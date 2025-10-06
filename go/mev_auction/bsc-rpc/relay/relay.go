package relay

import (
	"context"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/lru"
	"github.com/ethereum/go-ethereum/common/ms"
	"github.com/ethereum/go-ethereum/core/types"
	. "github.com/ethereum/go-ethereum/log/zap"
	pb "github.com/ethereum/go-ethereum/relay/protobuf"
	"github.com/ethereum/go-ethereum/rlp"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/keepalive"
	"time"
)

var SubServer = NewRelaySub()

type Sub struct {
	*ms.Server
	publicTxs *lru.Cache[common.Hash, struct{}]
}

func (sub *Sub) ServerName() string {
	return "relay-sub"
}

func (sub *Sub) MsgAction(ctx context.Context, msg interface{}, num int) (resp interface{}, err error) {
	return nil, nil
}

func (sub *Sub) ActionGoroutineNum() int {
	return 0
}

func (sub *Sub) Schedule() []ms.TimedTask {

	// BlockRazor relay endpoint address
	blzrelayEndPoint := "52.205.173.134:50051"

	// auth will be used to verify the credential
	auth := Authentication{
		"YWCVWDpexIVHRIDshlrH8Ohfv8gVCOEmHBbvH38DEkD3TIWWVGEiIUBYJUTnuqYb5ECf1NwssJoIB6UG4jnPmJCFMTFJC4G8",
	}

	// open gRPC connection to BlockRazor relay
	conn, err := grpc.NewClient(blzrelayEndPoint,
		grpc.WithInsecure(),
		grpc.WithPerRPCCredentials(&auth),
		grpc.WithWriteBufferSize(0),
		grpc.WithInitialConnWindowSize(128*1024),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{Time: 5 * time.Second, Timeout: 2 * time.Second, PermitWithoutStream: true}))

	if err != nil {
		panic(err)
	}

	// use the Gateway client connection interface
	client := pb.NewGatewayClient(conn)
	var stream pb.Gateway_NewTxsClient

	return []ms.TimedTask{{
		Task: func(num int) {
			if stream == nil {
				stream, err = client.NewTxs(context.Background(), &pb.TxsRequest{NodeValidation: false})
				if err != nil {
					time.Sleep(2 * time.Second)
					return
				}
			}

			reply, err := stream.Recv()
			if err != nil {
				Zap.Error("relay stream err", zap.Any("err", err))
				stream.CloseSend()
				stream = nil
				time.Sleep(2 * time.Second)
				return
			}

			tx := &types.Transaction{}
			err = rlp.DecodeBytes(reply.Tx.RawTx, tx)
			if err != nil {
				return
			}
			sub.publicTxs.Add(tx.Hash(), struct{}{})
		},
		Time: 0,
	}}
}

func (sub *Sub) SetServer(s *ms.Server) {
	sub.Server = s
}

func NewRelaySub() *Sub {
	s := &Sub{publicTxs: lru.NewCache[common.Hash, struct{}](20000)}
	ms.Init(s)

	s.Go()
	return s
}

func (sub *Sub) IsPublic(hash common.Hash) bool {
	return sub.publicTxs.Contains(hash)
}

// Authentication auth will be used to verify the credential
type Authentication struct {
	apiKey string
}

func (a *Authentication) GetRequestMetadata(context.Context, ...string) (map[string]string, error) {
	return map[string]string{"apiKey": a.apiKey}, nil
}

func (a *Authentication) RequireTransportSecurity() bool {
	return false
}
