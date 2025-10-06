package metrics

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestMyHistogram(t *testing.T) {
	mh := NewMyHistogram(1000, []float64{100, 300, 500, 700, 900})
	s := 0.0
	for i := 0; i < 1000; i++ {
		mh.Add(float64(i))
		s += float64(i)
	}
	snapshot := mh.Snapshot()
	ks, vs := snapshot.LessThenOrEquals(nil)
	assert.Equal(t, snapshot.Count(), int64(1000))
	assert.Equal(t, snapshot.MySum(), s)
	assert.NotNil(t, ks)
	assert.NotNil(t, vs)
	assert.Equal(t, len(ks), 5)
	assert.Equal(t, len(vs), 6)
	assert.Equal(t, ks, []float64{100, 300, 500, 700, 900})
	assert.Equal(t, vs, []int64{101, 200, 200, 200, 200, 99})
}
