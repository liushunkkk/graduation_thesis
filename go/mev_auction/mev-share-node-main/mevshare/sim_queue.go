package mevshare

import (
	"context"
	"encoding/json"
	"errors"
	"github.com/flashbots/mev-share-node/zap_logger"
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
	blockNumber := CurrentHeader.Number.Uint64()
	_ = q.queue.UpdateBlock(blockNumber)

	// 这些process会并发处理queue
	wg := q.queue.StartProcessLoop(ctx, process)

	wg.Add(1)
	go func() {
		defer wg.Done()

		back := backoff.NewExponentialBackOff()
		back.MaxInterval = 2 * time.Second
		back.MaxElapsedTime = 3 * time.Second

		ticker := time.NewTicker(100 * time.Millisecond)
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				err := backoff.Retry(func() error {
					blockNumber := CurrentHeader.Number.Uint64()
					return q.queue.UpdateBlock(blockNumber)
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
	//cancelCache       *RedisCancellationCache
	//replacementCache  *redis.ReplacementCache
	backgroundWg *sync.WaitGroup
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

	// Check if bundle was cancelled
	//cancelled, err := w.isBundleCancelled(ctx, &bundle)
	//if err != nil {
	//	// We don't return error here,  because we would consider this error as non-critical as our cancellations are "best effort".
	//	logger.Error("Failed to check if bundle was cancelled", zap.Error(err))
	//}
	//if cancelled {
	//	logger.Info("Bundle is not simulated because it was cancelled")
	//	return simqueue.ErrProcessUnrecoverable
	//}

	result, err := w.simulationBackend.SimulateBundle(ctx, &bundle, nil)
	if err != nil {
		logger.Error("Failed to simulate matched bundle", zap.Error(err))
		// we want to retry after such error
		return errors.Join(err, simqueue.ErrProcessWorkerError)
	}

	zap_logger.Zap.Info("Simulated bundle", zap.Any("hash", hash.Hex()), zap.Uint64("target_block", info.TargetBlock))

	logger.Info("Simulated bundle",
		zap.Bool("success", result.Success), zap.String("err_reason", result.Error),
		zap.String("gwei_eff_gas_price", formatUnits(result.MevGasPrice.ToInt(), "gwei")),
		zap.String("eth_profit", formatUnits(result.Profit.ToInt(), "eth")),
		zap.String("eth_refundable_value", formatUnits(result.RefundableValue.ToInt(), "eth")),
		zap.Uint64("gas_used", uint64(result.GasUsed)),
		zap.Uint64("state_block", uint64(result.StateBlock)),
		zap.String("exec_error", result.ExecError),
		zap.String("revert", result.Revert.String()),
		zap.Int("retries", info.Retries),
	)
	// mev-share-node knows that new block already arrived, but the node this worker connected to is lagging behind so we should retry
	if uint64(result.StateBlock) < info.TargetBlock-1 {
		logger.Warn("Bundle simulated on outdated block, retrying")
		return simqueue.ErrLaggingBlock
	}

	var isOldBundle bool
	//if bundle.ReplacementUUID != "" {
	//	rnonce, err := w.replacementCache.GetReplacementNonce(ctx, bundle.Metadata.Signer.String(), bundle.ReplacementUUID)
	//	if err != nil {
	//		// better send bundle and let builder decide if it is appropriate, don't fail here
	//		isOldBundle = false
	//		logger.Error("Failed to get replacement nonce", zap.Error(err))
	//	}
	//	if err == nil && rnonce > bundle.Metadata.ReplacementNonce {
	//		isOldBundle = true
	//	}
	//}

	//shouldCancel := bundle.ReplacementUUID != "" && !result.Success
	//if shouldCancel {
	//	logger.Info("Cancelling bundle", zap.String("replacement_uuid", bundle.ReplacementUUID))
	//	w.backgroundWg.Add(1)
	//	go func() {
	//		defer w.backgroundWg.Done()
	//		resCtx, cancel := context.WithTimeout(context.Background(), consumeSimulationTimeout)
	//		defer cancel()
	//		err = w.simRes.SimulatedBundle(resCtx, &bundle, result, info, shouldCancel, isOldBundle)
	//		if err != nil {
	//			w.log.Error("Failed to consume matched share bundle", zap.Error(err))
	//		}
	//	}()
	//	max := bundle.Inclusion.MaxBlock
	//	state := result.StateBlock
	//	// If state block is N, that means simulation for target block N+1 was tried
	//	if max != 0 && state != 0 && max > state+1 {
	//		return simqueue.ErrProcessScheduleNextBlock
	//	}
	//	return nil
	//}
	// Try to re-simulate bundle if it failed
	if !result.Success && isErrorRecoverable(result.Error) {
		max := bundle.Inclusion.MaxBlock
		state := result.StateBlock
		// If state block is N, that means simulation for target block N+1 was tried
		if max != 0 && state != 0 && max > state+1 {
			return simqueue.ErrProcessScheduleNextBlock
		}
	}

	w.backgroundWg.Add(1)
	go func() {
		defer w.backgroundWg.Done()
		resCtx, cancel := context.WithTimeout(context.Background(), consumeSimulationTimeout)
		defer cancel()
		err = w.simRes.SimulatedBundle(resCtx, &bundle, result, info, false, isOldBundle)
		if err != nil {
			w.log.Error("Failed to consume matched share bundle", zap.Error(err))
		}
	}()

	if !result.Success && !isErrorRecoverable(result.Error) {
		return simqueue.ErrProcessUnrecoverable
	}
	return nil
}

//func (w *SimulationWorker) isBundleCancelled(ctx context.Context, bundle *SendMevBundleArgs) (bool, error) {
//	ctx, cancel := context.WithTimeout(ctx, simCacheTimeout)
//	defer cancel()
//	if bundle.Metadata == nil {
//		w.log.Error("Bundle has no metadata, skipping cancel check")
//		return false, nil
//	}
//	res, err := w.cancelCache.IsCancelled(ctx, append([]common.Hash{bundle.Metadata.BundleHash}, bundle.Metadata.BodyHashes...))
//	if err != nil {
//		return false, err
//	}
//	return res, nil
//}

func isErrorRecoverable(message string) bool {
	return !strings.Contains(message, "nonce too low")
}
