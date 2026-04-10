package api

import (
	"container/heap"
	"fmt"
	"sync"
)

type queue []*Request

func (q queue) Len() int           { return len(q) }
func (q queue) Less(i, j int) bool { return q[i].priority > q[j].priority }
func (q queue) Swap(i, j int)      { q[i], q[j] = q[j], q[i] }
func (q *queue) Push(x any)        { *q = append(*q, x.(*Request)) }
func (q *queue) Pop() any {
	n := len(*q)
	x := (*q)[n-1]
	*q = (*q)[:n-1]
	return x
}

type PriorityQueue struct {
	mu     sync.Mutex
	buf    queue
	signal chan struct{}
}

func NewPriorityQueue() *PriorityQueue {
	return &PriorityQueue{
		buf:    queue{},
		signal: make(chan struct{}),
	}
}

func (p *PriorityQueue) Push(r *Request) {
	p.mu.Lock()
	defer p.mu.Unlock()
	heap.Push(&p.buf, r)
	select {
	case p.signal <- struct{}{}:
	default:
	}
}

func (p *PriorityQueue) Pop() (*Request, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.buf) == 0 {
		return nil, fmt.Errorf("priority queue is empty")
	}
	return heap.Pop(&p.buf).(*Request), nil
}
