package invalid_tx

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/ms"
	"github.com/syndtr/goleveldb/leveldb"
	"os"
	"sync"
	"time"
)

var Server *InvalidTxServer

const Dir = "invalid_tx_history/"

const Put = 0
const Delete = 1
const GetTxHash = 2

const Exist = "exist"

type TxMsg struct {
	Type   int // 0- put 1-delete
	TxHash common.Hash
}

type InvalidTxInfo struct {
	Time int64
}

func (tx InvalidTxInfo) Marshal() []byte {
	marshal, _ := json.Marshal(tx)
	return marshal
}

func UnMarshalToInvalidTxInfo(b []byte) *InvalidTxInfo {
	if len(b) == 0 {
		return nil
	}
	t := &InvalidTxInfo{}
	err := json.Unmarshal(b, t)
	if err != nil {
		return nil
	}
	return t
}

type InvalidTxServer struct {
	*ms.Server
	Buckets sync.Map
}

func (h *InvalidTxServer) ServerName() string {
	return "history"
}

func (h *InvalidTxServer) MsgAction(ctx context.Context, msg interface{}, num int) (resp interface{}, err error) {
	t := time.Now()
	var todayDB *leveldb.DB
	value, ok := h.Buckets.Load(h.GetToday(t))
	if !ok {
		todayDB, err = leveldb.OpenFile(h.GetToday(t), nil)
		if err != nil {
			fmt.Println("open leveldb failed")
			return nil, err
		}
		h.Buckets.Store(h.GetToday(t), todayDB)
	} else {
		todayDB = value.(*leveldb.DB)
	}
	txMsg := msg.(*TxMsg)
	if txMsg.Type == Put {
		todayDB.Put([]byte(txMsg.TxHash.Hex()), InvalidTxInfo{Time: time.Now().Unix()}.Marshal(), nil)
	} else if txMsg.Type == GetTxHash {
		key := txMsg.TxHash

		get, _ := todayDB.Get([]byte(key.Hex()), nil)
		if ret := UnMarshalToInvalidTxInfo(get); ret != nil {
			return ret, nil
		}

		yesterday, exist := h.Buckets.Load(h.GetYesterday(t))
		if exist {
			get, _ = yesterday.(*leveldb.DB).Get([]byte(key.Hex()), nil)
			if ret := UnMarshalToInvalidTxInfo(get); ret != nil {
				return ret, nil
			}
		}
	} else if txMsg.Type == Delete {
		todayDB.Delete([]byte(txMsg.TxHash.Hex()), nil)
		yesterday, exist := h.Buckets.Load(h.GetYesterday(t))
		if exist {
			yesterday.(*leveldb.DB).Delete([]byte(txMsg.TxHash.Hex()), nil)
		}
	}
	return nil, nil
}

func (h *InvalidTxServer) ActionGoroutineNum() int {
	return 1
}

func (h *InvalidTxServer) Schedule() []ms.TimedTask {
	return []ms.TimedTask{{
		Task: func(num int) {
			d := h.GetThreeDays(time.Now())
			v, ok := h.Buckets.Load(d)
			if ok {
				v.(*leveldb.DB).Close()
				h.Buckets.Delete(d)
			}
			os.RemoveAll(d)
			fmt.Printf("delete leveldb[%s] successfully\n", d)
		},
		Time: 1 * time.Hour,
	}}
}

func (h *InvalidTxServer) SetServer(s *ms.Server) {
	h.Server = s
}

func New() *InvalidTxServer {
	h := &InvalidTxServer{}
	ms.Init(h)
	return h
}

func (h *InvalidTxServer) Start() {
	h.Go()
}

func (h *InvalidTxServer) Destroy() {
	h.Stop()
}

func (h *InvalidTxServer) GetToday(t time.Time) string {
	return Dir + t.Format("2006-01-02")
}

func (h *InvalidTxServer) GetYesterday(t time.Time) string {
	return Dir + t.Add(-24*time.Hour).Format("2006-01-02")
}

func (h *InvalidTxServer) GetThreeDays(t time.Time) string {
	return Dir + t.Add(-24*time.Hour*3).Format("2006-01-02")
}

func (h *InvalidTxServer) Put(txTxHash common.Hash) {
	h.PushMsgToServer(context.Background(), &TxMsg{
		Type:   Put,
		TxHash: txTxHash,
	})
}

func (h *InvalidTxServer) Delete(TxHash common.Hash) {
	h.PushMsgToServer(context.Background(), &TxMsg{
		Type:   Delete,
		TxHash: TxHash,
	})
}

func (h *InvalidTxServer) Get(TxHash common.Hash) *InvalidTxInfo {
	resp, _ := h.PostMsgToServer(context.Background(), &TxMsg{
		Type:   GetTxHash,
		TxHash: TxHash,
	})
	if v, ok := resp.(*InvalidTxInfo); ok {
		return v
	}
	return nil
}
