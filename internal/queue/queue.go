// Package queue publishes ingestion jobs to Kafka.
//
// Ingestion is asynchronous because it is slow and bursty: chunking and
// embedding a large document takes seconds to minutes, and an upload endpoint
// that waits for it will time out on exactly the documents people care about.
// The API stores the raw text, publishes a job and returns immediately; the
// Python workers consume at whatever rate the embedding quota allows.
package queue

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/segmentio/kafka-go"
)

// IngestJob is the message body. It deliberately carries no document text —
// only the id. Kafka's default message cap is 1MB and documents routinely
// exceed it, so the payload is a pointer into Postgres rather than the content.
type IngestJob struct {
	DocumentID uuid.UUID `json:"document_id"`
	SourceURI  string    `json:"source_uri"`
	Title      string    `json:"title"`
	Attempt    int       `json:"attempt"`
	EnqueuedAt time.Time `json:"enqueued_at"`
}

type Producer struct{ w *kafka.Writer }

func NewProducer(brokers []string) *Producer {
	return &Producer{w: &kafka.Writer{
		Addr:         kafka.TCP(brokers...),
		Balancer:     &kafka.Hash{},
		RequiredAcks: kafka.RequireAll,
		BatchTimeout: 50 * time.Millisecond,
		MaxAttempts:  5,
	}}
}

func (p *Producer) Close() error { return p.w.Close() }

// Publish keys by document id so that repeated jobs for one document stay
// ordered on a single partition — a re-ingest must not race the first ingest.
func (p *Producer) Publish(ctx context.Context, topic string, job IngestJob) error {
	body, err := json.Marshal(job)
	if err != nil {
		return err
	}
	return p.w.WriteMessages(ctx, kafka.Message{
		Topic: topic,
		Key:   []byte(job.DocumentID.String()),
		Value: body,
		Time:  time.Now(),
	})
}

// EnsureTopics pre-creates topics; managed clusters usually disable
// auto-creation and relying on it makes the first upload flaky.
func EnsureTopics(ctx context.Context, brokers []string, topics []string, partitions int) error {
	conn, err := kafka.DialContext(ctx, "tcp", brokers[0])
	if err != nil {
		return err
	}
	defer conn.Close()
	controller, err := conn.Controller()
	if err != nil {
		return err
	}
	cc, err := kafka.DialContext(ctx, "tcp", fmt.Sprintf("%s:%d", controller.Host, controller.Port))
	if err != nil {
		return err
	}
	defer cc.Close()

	cfgs := make([]kafka.TopicConfig, 0, len(topics))
	for _, t := range topics {
		cfgs = append(cfgs, kafka.TopicConfig{Topic: t, NumPartitions: partitions, ReplicationFactor: 1})
	}
	return cc.CreateTopics(cfgs...)
}
