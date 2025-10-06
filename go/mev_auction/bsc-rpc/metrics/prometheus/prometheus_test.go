package prometheus

import (
	"bytes"
	"fmt"
	"github.com/ethereum/go-ethereum/metrics"
	"net/http"
	"testing"
)

type testResponseWriter struct {
	buf *bytes.Buffer
}

func (t testResponseWriter) Header() http.Header {
	return http.Header{}
}

func (t testResponseWriter) Write(i []byte) (int, error) {
	return t.buf.Write(i)
}

func (t testResponseWriter) WriteHeader(statusCode int) {
}

func TestMyHistogramMetrics(t *testing.T) {
	reg := metrics.NewRegistry()
	mh := metrics.GetOrRegisterMyHistogram("test/myhistogram", reg, 1000, []float64{100, 300, 500, 700, 900})
	s := 0.0
	for i := 0; i < 1010; i++ {
		mh.Add(float64(i))
		s += float64(i)
	}
	buffer := &testResponseWriter{
		buf: bytes.NewBuffer(nil),
	}
	Handler(reg).ServeHTTP(buffer, nil)
	fmt.Println(buffer.buf.String())
}
