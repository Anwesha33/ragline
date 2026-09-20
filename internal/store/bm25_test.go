package store

import (
	"context"
	"os"
	"testing"

	"github.com/google/uuid"
)

// These run against a real Postgres, because the thing under test is a set of
// SQL triggers. Skipped unless RAGLINE_TEST_DSN is set:
//
//	RAGLINE_TEST_DSN='postgres://ragline:ragline@localhost:5433/ragline?sslmode=disable' \
//	  go test ./internal/store -run BM25
func bm25TestStore(t *testing.T) *Store {
	t.Helper()
	dsn := os.Getenv("RAGLINE_TEST_DSN")
	if dsn == "" {
		t.Skip("RAGLINE_TEST_DSN not set")
	}
	st, err := Open(context.Background(), dsn)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if err := st.Migrate(context.Background()); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	t.Cleanup(st.Close)
	return st
}

// seedDoc inserts a document and its chunks, and removes them afterwards so the
// corpus statistics return to where they started.
func seedDoc(t *testing.T, st *Store, chunks []struct{ heading, body string }) uuid.UUID {
	t.Helper()
	ctx := context.Background()
	id := uuid.New()
	_, err := st.pool.Exec(ctx,
		`INSERT INTO documents (id, source_uri, title, content_hash, status)
		 VALUES ($1,'test://bm25','bm25 fixture',$2,'ready')`, id, id.String())
	if err != nil {
		t.Fatalf("insert document: %v", err)
	}
	for i, c := range chunks {
		if _, err := st.pool.Exec(ctx,
			`INSERT INTO chunks (document_id, ordinal, heading, content) VALUES ($1,$2,$3,$4)`,
			id, i, c.heading, c.body); err != nil {
			t.Fatalf("insert chunk: %v", err)
		}
	}
	t.Cleanup(func() {
		st.pool.Exec(context.Background(), `DELETE FROM documents WHERE id = $1`, id)
	})
	return id
}

func corpusStats(t *testing.T, st *Store) (chunkCount, totalLen int64) {
	t.Helper()
	err := st.pool.QueryRow(context.Background(),
		`SELECT chunk_count, total_len FROM corpus_stats WHERE only_row`).Scan(&chunkCount, &totalLen)
	if err != nil {
		t.Fatalf("corpus_stats: %v", err)
	}
	return
}

// The statistics must be written in the same transaction as the chunk, so they
// are correct the instant the chunk is searchable.
func TestBM25StatsMaintainedOnInsertAndDelete(t *testing.T) {
	st := bm25TestStore(t)
	ctx := context.Background()

	beforeCount, beforeLen := corpusStats(t, st)

	docID := seedDoc(t, st, []struct{ heading, body string }{
		{"Refunds", "a refund takes five days"},
		{"Chargebacks", "a chargeback is a dispute raised with the card issuer"},
	})

	afterCount, afterLen := corpusStats(t, st)
	if afterCount != beforeCount+2 {
		t.Errorf("chunk_count = %d, want %d", afterCount, beforeCount+2)
	}
	if afterLen <= beforeLen {
		t.Errorf("total_len did not grow: %d -> %d", beforeLen, afterLen)
	}

	// Every chunk must have both term rows and a length.
	var orphans int
	if err := st.pool.QueryRow(ctx, `
		SELECT COUNT(*) FROM chunks c
		WHERE c.document_id = $1
		  AND (NOT EXISTS (SELECT 1 FROM chunk_terms t WHERE t.chunk_id = c.id)
		    OR NOT EXISTS (SELECT 1 FROM chunk_stats s WHERE s.chunk_id = c.id))`,
		docID).Scan(&orphans); err != nil {
		t.Fatal(err)
	}
	if orphans != 0 {
		t.Errorf("%d chunks have no BM25 statistics", orphans)
	}

	// Deleting the document must return the corpus to where it started.
	if _, err := st.pool.Exec(ctx, `DELETE FROM documents WHERE id = $1`, docID); err != nil {
		t.Fatal(err)
	}
	finalCount, finalLen := corpusStats(t, st)
	if finalCount != beforeCount {
		t.Errorf("chunk_count after delete = %d, want %d", finalCount, beforeCount)
	}
	if finalLen != beforeLen {
		t.Errorf("total_len after delete = %d, want %d", finalLen, beforeLen)
	}
}

// df must count chunks, not occurrences: a term appearing three times in one
// chunk has a document frequency of one.
func TestBM25DocumentFrequencyCountsChunksNotOccurrences(t *testing.T) {
	st := bm25TestStore(t)
	ctx := context.Background()

	seedDoc(t, st, []struct{ heading, body string }{
		{"", "zzqterm zzqterm zzqterm"},
	})

	var df int64
	if err := st.pool.QueryRow(ctx,
		`SELECT df FROM corpus_terms WHERE term = 'zzqterm'`).Scan(&df); err != nil {
		t.Fatalf("corpus_terms: %v", err)
	}
	if df != 1 {
		t.Errorf("df = %d, want 1", df)
	}

	var tf int
	if err := st.pool.QueryRow(ctx,
		`SELECT tf FROM chunk_terms WHERE term = 'zzqterm'`).Scan(&tf); err != nil {
		t.Fatalf("chunk_terms: %v", err)
	}
	if tf != 3 {
		t.Errorf("tf = %d, want 3", tf)
	}
}

// A term in the heading is weighted 'A' and must count for more than the same
// term in the body, preserving the property the weighted tsvector already had.
func TestBM25HeadingTermsWeighHeavier(t *testing.T) {
	st := bm25TestStore(t)
	ctx := context.Background()

	seedDoc(t, st, []struct{ heading, body string }{
		{"zzheadterm", "filler text about nothing in particular"},
		{"", "zzheadterm"},
	})

	// The stored term is the stemmed lexeme, not the word as written — the
	// statistics are derived from the tsvector, so they inherit its analysis.
	rows, err := st.pool.Query(ctx,
		`SELECT tf FROM chunk_terms WHERE term = 'zzheadterm' ORDER BY tf DESC`)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	var tfs []int
	for rows.Next() {
		var tf int
		if err := rows.Scan(&tf); err != nil {
			t.Fatal(err)
		}
		tfs = append(tfs, tf)
	}
	if len(tfs) != 2 {
		t.Fatalf("got %d rows, want 2: %v", len(tfs), tfs)
	}
	if tfs[0] <= tfs[1] {
		t.Errorf("heading tf %d should exceed body tf %d", tfs[0], tfs[1])
	}
}

// The property that distinguishes BM25 from coverage density: a rare term is
// worth more than a common one. ts_rank_cd has no corpus-wide notion of
// rarity at all.
func TestBM25PrefersTheRareTerm(t *testing.T) {
	st := bm25TestStore(t)
	ctx := context.Background()

	// "common" appears in every chunk; "zzrare" in exactly one.
	fixture := []struct{ heading, body string }{
		{"", "common filler one"},
		{"", "common filler two"},
		{"", "common filler three"},
		{"", "common filler four"},
		{"", "common zzrare five"},
	}
	seedDoc(t, st, fixture)

	var commonDF, rareDF int64
	if err := st.pool.QueryRow(ctx, `SELECT df FROM corpus_terms WHERE term='common'`).Scan(&commonDF); err != nil {
		t.Fatal(err)
	}
	if err := st.pool.QueryRow(ctx, `SELECT df FROM corpus_terms WHERE term='zzrare'`).Scan(&rareDF); err != nil {
		t.Fatal(err)
	}
	if commonDF <= rareDF {
		t.Fatalf("fixture is wrong: common df=%d rare df=%d", commonDF, rareDF)
	}

	// Ask for both terms; the chunk holding the rare one must rank first.
	st.KeywordRanking = KeywordBM25
	got, err := st.KeywordSearch(ctx, "common zzrare", 5)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("no results")
	}
	if !containsWord(got[0].Content, "zzrare") {
		t.Errorf("top hit %q does not contain the rare term; BM25 IDF is not being applied", got[0].Content)
	}
}

// Both rankers must return results for the same query; the point of keeping
// ts_rank is that it is a working control arm, not a broken path.
func TestBM25AndTsRankBothReturnResults(t *testing.T) {
	st := bm25TestStore(t)
	ctx := context.Background()
	seedDoc(t, st, []struct{ heading, body string }{
		{"Refunds", "a refund takes five working days to reach the card"},
		{"Disputes", "a chargeback is raised with the issuer"},
	})

	for _, mode := range []KeywordRanking{KeywordBM25, KeywordTsRank} {
		st.KeywordRanking = mode
		got, err := st.KeywordSearch(ctx, "refund card", 5)
		if err != nil {
			t.Fatalf("%s: %v", mode, err)
		}
		if len(got) == 0 {
			t.Errorf("%s returned no results", mode)
		}
	}
}

func containsWord(haystack, needle string) bool {
	for _, f := range splitFields(haystack) {
		if f == needle {
			return true
		}
	}
	return false
}

func splitFields(s string) []string {
	var out []string
	cur := ""
	for _, r := range s {
		if r == ' ' || r == '\n' || r == '\t' {
			if cur != "" {
				out = append(out, cur)
				cur = ""
			}
			continue
		}
		cur += string(r)
	}
	if cur != "" {
		out = append(out, cur)
	}
	return out
}

// An existing corpus was ingested before the triggers existed, so its chunks
// are searchable by tsvector and invisible to BM25 until the statistics are
// backfilled. That is a silent retrieval regression on upgrade, so Migrate
// backfills — and must be idempotent, because Migrate runs on every start.
func TestBM25BackfillRestoresStatisticsForAnExistingCorpus(t *testing.T) {
	st := bm25TestStore(t)
	ctx := context.Background()

	seedDoc(t, st, []struct{ heading, body string }{
		{"Refunds", "a refund takes five working days"},
		{"Disputes", "a chargeback is raised with the issuer"},
	})

	want, wantLen := corpusStats(t, st)
	var wantTerms int64
	if err := st.pool.QueryRow(ctx, `SELECT COUNT(*) FROM chunk_terms`).Scan(&wantTerms); err != nil {
		t.Fatal(err)
	}

	// Simulate the pre-BM25 state: chunks present, statistics absent.
	for _, stmt := range []string{
		`DELETE FROM chunk_terms`,
		`DELETE FROM chunk_stats`,
		`DELETE FROM corpus_terms`,
		`UPDATE corpus_stats SET chunk_count = 0, total_len = 0 WHERE only_row`,
	} {
		if _, err := st.pool.Exec(ctx, stmt); err != nil {
			t.Fatalf("%s: %v", stmt, err)
		}
	}

	if err := st.Migrate(ctx); err != nil {
		t.Fatalf("migrate: %v", err)
	}

	got, gotLen := corpusStats(t, st)
	if got != want || gotLen != wantLen {
		t.Errorf("after backfill: chunk_count=%d total_len=%d, want %d/%d", got, gotLen, want, wantLen)
	}
	var gotTerms int64
	if err := st.pool.QueryRow(ctx, `SELECT COUNT(*) FROM chunk_terms`).Scan(&gotTerms); err != nil {
		t.Fatal(err)
	}
	if gotTerms != wantTerms {
		t.Errorf("chunk_terms rows = %d, want %d", gotTerms, wantTerms)
	}

	// Idempotent: running it again must not double anything.
	if err := st.Migrate(ctx); err != nil {
		t.Fatalf("second migrate: %v", err)
	}
	again, againLen := corpusStats(t, st)
	if again != got || againLen != gotLen {
		t.Errorf("second migrate changed the statistics: %d/%d -> %d/%d", got, gotLen, again, againLen)
	}
}
