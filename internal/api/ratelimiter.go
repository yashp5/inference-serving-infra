package api

import (
	"sync"
	"time"
)

type RateLimiter interface {
	allow(userId string) bool
}

func NewRateLimiter(limiterType string, N int, D int) RateLimiter {
	switch limiterType {
	default:
		return NewTokenBucketRateLimiter(N, D)
	}
}

type Bucket struct {
	tokens        float64
	ratePerMs     float64
	lastTimestamp time.Time
}

type TokenBucketRateLimiter struct {
	mu      sync.Mutex
	clients map[string]*Bucket
	N       int
	D       int
}

func NewTokenBucketRateLimiter(N int, D int) *TokenBucketRateLimiter {
	return &TokenBucketRateLimiter{
		clients: make(map[string]*Bucket),
		N:       N,
		D:       D,
	}
}

func (r *TokenBucketRateLimiter) allow(userId string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	b, ok := r.clients[userId]
	if !ok {
		b = &Bucket{
			tokens:        float64(r.N),
			ratePerMs:     float64(r.N) / float64(r.D),
			lastTimestamp: time.Now(),
		}
		r.clients[userId] = b
	}

	now := time.Now()
	accrual := b.ratePerMs * (float64(now.Sub(b.lastTimestamp)))
	b.tokens = min(b.tokens+accrual, float64(r.N))
	b.lastTimestamp = now

	if b.tokens-1.0 < 0 {
		return false
	}

	b.tokens = max(b.tokens-1.0, 0.0)
	return true
}
