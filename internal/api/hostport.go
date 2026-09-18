package api

import "net"

func splitHostPort(addr string) (string, string, error) { return net.SplitHostPort(addr) }
