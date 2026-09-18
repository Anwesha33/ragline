package cache

import "strconv"

func parseFloat(s string) (float64, error) { return strconv.ParseFloat(s, 64) }
func itoa(n int) string                    { return strconv.Itoa(n) }
