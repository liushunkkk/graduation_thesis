package mevshare

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"time"

	"github.com/cenkalti/backoff/v4"
	"github.com/ethereum/go-ethereum/common"
	"github.com/flashbots/mev-share-node/simqueue"
	"go.uber.org/zap"
	"golang.org/x/time/rate"
)

var (
	consumeSimulationTimeout = 5 * time.Second
	simCacheTimeout          = 1 * time.Second
)

type SimQueue struct {
	log            *zap.Logger
	queue          simqueue.Queue
	workers        []SimulationWorker
	workersPerNode int
}

// NewQueue 创建队列，初始化worker
// 默认 workersPerNode = 2
func NewQueue(
	log *zap.Logger, queue simqueue.Queue, sim []SimulationBackend, simRes SimulationResult,
	workersPerNode int, backgroundWg *sync.WaitGroup,
) *SimQueue {
	log = log.Named("queue")
	q := &SimQueue{
		log:            log,
		queue:          queue,
		workers:        make([]SimulationWorker, 0, len(sim)),
		workersPerNode: workersPerNode,
	}

	for i := range sim {
		worker := SimulationWorker{
			log:               log.Named("worker").With(zap.Int("worker-id", i)),
			simulationBackend: sim[i],
			simRes:            simRes,
			backgroundWg:      backgroundWg,
		}
		q.workers = append(q.workers, worker)
	}
	return q
}

// Start 会启动模拟bundle的队列
// q.queue.StartProcessLoop(ctx, process) 会开启队列处理的 loop
func (q *SimQueue) Start(ctx context.Context) *sync.WaitGroup {
	process := make([]simqueue.ProcessFunc, 0, len(q.workers)*q.workersPerNode)
	for i := range q.workers {
		if q.workersPerNode > 1 {
			// len(workers) = q.workersPerNode
			workers := simqueue.MultipleWorkers(q.workers[i].Process, q.workersPerNode, rate.Inf, 1)
			process = append(process, workers...)
		} else {
			process = append(process, q.workers[i].Process)
		}
	}

	// 这些process会并发处理queue
	wg := q.queue.StartProcessLoop(ctx, process)

	wg.Add(1)
	go func() {
		defer wg.Done()

		back := backoff.NewExponentialBackOff()
		back.MaxInterval = 3 * time.Second
		back.MaxElapsedTime = 12 * time.Second

		ticker := time.NewTicker(100 * time.Millisecond)
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				err := backoff.Retry(func() error {
					return nil
				}, back)
				if err != nil {
					q.log.Error("Failed to update block number", zap.Error(err))
				}
			}
		}
	}()
	return wg
}

func (q *SimQueue) ScheduleBundleSimulation(ctx context.Context, bundle *SendMevBundleArgs, highPriority bool) error {
	data, err := json.Marshal(bundle)
	if err != nil {
		return err
	}
	return q.queue.Push(ctx, data, highPriority, uint64(bundle.Inclusion.BlockNumber), uint64(bundle.Inclusion.MaxBlock))
}

type SimulationWorker struct {
	log               *zap.Logger
	simulationBackend SimulationBackend
	simRes            SimulationResult
	backgroundWg      *sync.WaitGroup
}

// Process 是模拟bundle队列的处理器，
// 他会调用 w.simRes.SimulatedBundle 函数判断模拟的结果对不对
// 如果对会调用 s.builders.SendBundle 将bundle发送给builder
func (w *SimulationWorker) Process(ctx context.Context, data []byte, info simqueue.QueueItemInfo) (err error) {
	var bundle SendMevBundleArgs
	err = json.Unmarshal(data, &bundle)
	if err != nil {
		w.log.Error("Failed to unmarshal bundle simulation data", zap.Error(err))
		return err
	}

	var hash common.Hash
	if bundle.Metadata != nil {
		hash = bundle.Metadata.BundleHash
	}
	logger := w.log.With(zap.String("bundle", hash.Hex()), zap.Uint64("target_block", info.TargetBlock))

	result, err := w.simulationBackend.SimulateBundle(ctx, &bundle, nil)
	if err != nil {
		logger.Error("Failed to simulate matched bundle", zap.Error(err))
		// we want to retry after such error
		return errors.Join(err, simqueue.ErrProcessWorkerError)
	}

	logger.Info("Simulated bundle",
		zap.Bool("success", result.Success), zap.String("err_reason", result.Error),
	)

	w.backgroundWg.Add(1)
	go func() {
		defer w.backgroundWg.Done()
		resCtx, cancel := context.WithTimeout(context.Background(), consumeSimulationTimeout)
		defer cancel()
		err = w.simRes.SimulatedBundle(resCtx, &bundle, result, info, false, false)
		if err != nil {
			w.log.Error("Failed to consume matched share bundle", zap.Error(err))
		}
	}()

	if !result.Success && !isErrorRecoverable(result.Error) {
		return simqueue.ErrProcessUnrecoverable
	}
	return nil
}

func isErrorRecoverable(message string) bool {
	return !strings.Contains(message, "nonce too low")
}
