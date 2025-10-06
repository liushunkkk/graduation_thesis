package push

import (
	"context"
	"github.com/agiledragon/gomonkey/v2"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/ms"
	"github.com/ethereum/go-ethereum/push/blockrazor"
	"github.com/ethereum/go-ethereum/push/club48"
	"github.com/ethereum/go-ethereum/push/define"
	"github.com/ethereum/go-ethereum/push/smith"
	"github.com/smartystreets/goconvey/convey"
	"reflect"
	"testing"
	"time"
)

//func TestNewBuilderServer(t *testing.T) {
//	convey.Convey("qiguai", t, func() {
//		buildServer := NewBuilderServer()
//		buildServer.Start()
//		time.Sleep(1 * time.Second)
//		b := &types.Bundle{PrivacyBuilder: []string{"blockrazor", "48club"}, BroadcastBuilder: []string{"blockrazor", "48club"}}
//		patch := gomonkey.ApplyMethod(reflect.TypeOf(b), "GenBuilderReq", func(bundle *types.Bundle, header *types.Header) (*define.Param, *txv2.BundleSaveRequest) {
//			return &define.Param{
//				Txs:               nil,
//				MaxBlockNumber:    100,
//				BlockNumber:       "14839",
//				MinTimestamp:      0,
//				MaxTimestamp:      0,
//				RevertingTxHashes: nil,
//				BlockrazorOpts:    nil,
//			}, nil
//		})
//		defer patch.Reset()
//
//		buildServer.Send(&types.Header{Number: big.NewInt(100)}, b, 100)
//		time.Sleep(1 * time.Second)
//	})
//}

func TestNewBuilderServer(t *testing.T) {
	convey.Convey("test", t, func() {
		Builders = map[string]Builder{
			"blockrazor": blockrazor.NewBlockRazor(),
			"48club":     club48.NewClub48(),
			"smith":      smith.NewSmith(),
		}

		blockrazorPrivateTxCount := 0
		club48PrivateTxCount := 0
		smithPrivateTxCount := 0

		blockrazorBundleCount := 0
		club48BundleCount := 0
		smithBundleCount := 0

		blockrazorMethod := gomonkey.ApplyMethod(reflect.TypeOf(&blockrazor.BlockRazor{}), "SendRawPrivateTransaction", func() {
			blockrazorPrivateTxCount++
		})
		defer blockrazorMethod.Reset()

		method48 := gomonkey.ApplyMethod(reflect.TypeOf(&club48.Club48{}), "SendRawPrivateTransaction", func() {
			club48PrivateTxCount++
		})
		defer method48.Reset()

		simthMethod := gomonkey.ApplyMethod(reflect.TypeOf(&smith.Smith{}), "SendRawPrivateTransaction", func() {
			smithPrivateTxCount++
		})
		defer simthMethod.Reset()

		blockrazorMethod1 := gomonkey.ApplyMethod(reflect.TypeOf(&blockrazor.BlockRazor{}), "SendBundle", func() {
			blockrazorBundleCount++
		})
		defer blockrazorMethod1.Reset()

		method481 := gomonkey.ApplyMethod(reflect.TypeOf(&club48.Club48{}), "SendBundle", func() {
			club48BundleCount++
		})
		defer method481.Reset()

		simthMethod1 := gomonkey.ApplyMethod(reflect.TypeOf(&smith.Smith{}), "SendBundle", func() {
			smithBundleCount++
		})
		defer simthMethod1.Reset()

		buildServer := NewBuilderServer()
		buildServer.Start()

		for name, _ := range Builders {
			bp := &define.BuilderParam{BundleHash: common.Hash{}, Counter: 1, Param: &define.Param{Txs: []string{""}}}
			ms.PushMsgToServer(context.Background(), name, bp)
		}
		time.Sleep(5 * time.Millisecond)

		convey.So(blockrazorPrivateTxCount, convey.ShouldEqual, 0)
		convey.So(club48PrivateTxCount, convey.ShouldEqual, 0)
		convey.So(smithPrivateTxCount, convey.ShouldEqual, 0)

		convey.So(blockrazorBundleCount, convey.ShouldEqual, 1)
		convey.So(club48BundleCount, convey.ShouldEqual, 1)
		convey.So(smithBundleCount, convey.ShouldEqual, 1)

		blockrazorPrivateTxCount = 0
		club48PrivateTxCount = 0
		smithPrivateTxCount = 0

		blockrazorBundleCount = 0
		club48BundleCount = 0
		smithBundleCount = 0

		for name, _ := range Builders {
			bp := &define.BuilderParam{BundleHash: common.Hash{}, Counter: 0, Param: &define.Param{Txs: []string{""}}}
			ms.PushMsgToServer(context.Background(), name, bp)
		}
		time.Sleep(5 * time.Millisecond)

		convey.So(blockrazorPrivateTxCount, convey.ShouldEqual, 1)
		convey.So(club48PrivateTxCount, convey.ShouldEqual, 1)
		convey.So(smithPrivateTxCount, convey.ShouldEqual, 0)

		convey.So(blockrazorBundleCount, convey.ShouldEqual, 0)
		convey.So(club48BundleCount, convey.ShouldEqual, 1)
		convey.So(smithBundleCount, convey.ShouldEqual, 1)

		blockrazorPrivateTxCount = 0
		club48PrivateTxCount = 0
		smithPrivateTxCount = 0

		blockrazorBundleCount = 0
		club48BundleCount = 0
		smithBundleCount = 0

		for name, _ := range Builders {
			bp := &define.BuilderParam{BundleHash: common.Hash{}, Counter: 0, Param: &define.Param{Txs: []string{""}}}
			ms.PushMsgToServer(context.Background(), name, bp)
		}
		time.Sleep(5 * time.Millisecond)

		convey.So(blockrazorPrivateTxCount, convey.ShouldEqual, 1)
		convey.So(club48PrivateTxCount, convey.ShouldEqual, 1)
		convey.So(smithPrivateTxCount, convey.ShouldEqual, 0)

		convey.So(blockrazorBundleCount, convey.ShouldEqual, 0)
		convey.So(club48BundleCount, convey.ShouldEqual, 1)
		convey.So(smithBundleCount, convey.ShouldEqual, 1)

	})
}
