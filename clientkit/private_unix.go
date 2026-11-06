//go:build !windows

package clientkit

import (
	"fmt"
	"os"
)

func encodePrivate(data []byte) ([]byte, error) { return data, nil }
func replacePrivateFile(from, to string) error  { return os.Rename(from, to) }
func decodePrivate(data []byte, info os.FileInfo) ([]byte, error) {
	if info.Mode().Perm()&0077 != 0 {
		return nil, fmt.Errorf("credential file must be private (chmod 600)")
	}
	return data, nil
}
