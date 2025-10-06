package metrics

import (
	"sort"
	"sync"
)

type MyHistogramSnapshot interface {
	Count() int64
	LessThenOrEqual(float64) int64
	LessThenOrEquals([]float64) ([]float64, []int64)
	MySum() float64
}

// MyHistogram tracks how many values fall into specific ranges (buckets),
// along with the total number of values and their sum.
type MyHistogram interface {
	Clear()
	Add(float64)
	Snapshot() MyHistogramSnapshot
}

// GetOrRegisterMyHistogram returns an existing MyHistogram or constructs and registers
// a new StandardMyHistogram.
func GetOrRegisterMyHistogram(name string, r Registry, reservoirSize int, buckets []float64) MyHistogram {
	if nil == r {
		r = DefaultRegistry
	}
	return r.GetOrRegister(name, NewMyHistogram(reservoirSize, buckets)).(MyHistogram)
}

// GetOrRegisterMyHistogramForced returns an existing MyHistogram or constructs and registers a
// new MyHistogram no matter the global switch is enabled or not.
// Be sure to unregister the MyHistogram from the registry once it is of no use to
// allow for garbage collection.
func GetOrRegisterMyHistogramForced(name string, r Registry, reservoirSize int, buckets []float64) MyHistogram {
	if nil == r {
		r = DefaultRegistry
	}
	return r.GetOrRegister(name, NewMyHistogramForced(reservoirSize, buckets)).(MyHistogram)
}

// NewMyHistogram constructs a new StandardMyHistogram.
func NewMyHistogram(reservoirSize int, buckets []float64) MyHistogram {
	if !Enabled {
		return NilMyHistogram{}
	}
	smh := new(StandardMyHistogram)
	sort.Float64s(buckets)
	smh.reservoirSize = reservoirSize
	smh.buckets = buckets
	return smh
}

// NewMyHistogramForced constructs a new StandardMyHistogram and returns it no matter if
// the global switch is enabled or not.
func NewMyHistogramForced(reservoirSize int, buckets []float64) MyHistogram {
	smh := new(StandardMyHistogram)
	sort.Float64s(buckets)
	smh.buckets = buckets
	smh.reservoirSize = reservoirSize
	return smh
}

// NewRegisteredMyHistogram constructs and registers a new StandardMyHistogram.
func NewRegisteredMyHistogram(name string, r Registry, reservoirSize int, buckets []float64) MyHistogram {
	c := NewMyHistogram(reservoirSize, buckets)
	if nil == r {
		r = DefaultRegistry
	}
	r.Register(name, c)
	return c
}

// NewRegisteredMyHistogramForced constructs and registers a new StandardMyHistogram
// and launches a goroutine no matter the global switch is enabled or not.
// Be sure to unregister the MyHistogram from the registry once it is of no use to
// allow for garbage collection.
func NewRegisteredMyHistogramForced(name string, r Registry, reservoirSize int, buckets []float64) MyHistogram {
	c := NewMyHistogramForced(reservoirSize, buckets)
	if nil == r {
		r = DefaultRegistry
	}
	r.Register(name, c)
	return c
}

// myHistogramSnapshot is a read-only copy of another MyHistogram.
type myHistogramSnapshot struct {
	sum     float64
	count   int64
	values  []float64
	buckets []float64
}

// Count returns the count at the time the snapshot was taken.
func (s myHistogramSnapshot) Count() int64 { return s.count }

func (s myHistogramSnapshot) LessThenOrEqual(f float64) int64 {
	count := 0
	for _, v := range s.values {
		if v <= f {
			count++
		}
	}
	return int64(count)
}

func (s myHistogramSnapshot) LessThenOrEquals(buckets []float64) ([]float64, []int64) {
	if buckets == nil || len(buckets) == 0 {
		buckets = s.buckets
	}
	if buckets == nil || len(buckets) == 0 {
		buckets = []float64{5, 10, 20, 40, 80}
	}
	res := make([]int64, len(buckets)+1)
	for _, v := range s.values {
		if v > buckets[len(buckets)-1] {
			res[len(buckets)] += 1
			continue
		}
		for bi, b := range buckets {
			if bi == 0 {
				if v <= b {
					res[bi] += 1
				}
			} else {
				if v <= b && v > buckets[bi-1] {
					res[bi] += 1
				}
			}
		}
	}
	return buckets, res
}

func (s myHistogramSnapshot) MySum() float64 { return s.sum }

// NilMyHistogram is a no-op MyHistogram.
type NilMyHistogram struct{}

func (NilMyHistogram) Clear()                        {}
func (NilMyHistogram) Add(float64)                   {}
func (NilMyHistogram) Snapshot() MyHistogramSnapshot { return (*emptySnapshot)(nil) }

// StandardMyHistogram is the standard implementation of a MyHistogram.
type StandardMyHistogram struct {
	count         int64
	mutex         sync.Mutex
	reservoirSize int
	values        []float64
	buckets       []float64
}

// Clear sets the MyHistogram to zero.
func (c *StandardMyHistogram) Clear() {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.count = 0
	c.values = make([]float64, 0, c.reservoirSize)
}

// Add adds a value into the MyHistogram.
func (c *StandardMyHistogram) Add(v float64) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.count++
	if len(c.values) < c.reservoirSize {
		c.values = append(c.values, v)
	} else {
		n := len(c.values)
		h := c.values[1:n]
		h = append(h, v)
		c.values = h
	}
}

// Snapshot returns a read-only copy of the MyHistogram.
func (c *StandardMyHistogram) Snapshot() MyHistogramSnapshot {
	c.mutex.Lock()
	values := make([]float64, len(c.values))
	copy(values, c.values)
	buckets := make([]float64, len(c.buckets))
	copy(buckets, c.buckets)
	count := c.count
	c.mutex.Unlock()
	s := 0.0
	for _, v := range values {
		s += v
	}
	return myHistogramSnapshot{
		sum:     s,
		count:   count,
		values:  values,
		buckets: buckets,
	}
}
