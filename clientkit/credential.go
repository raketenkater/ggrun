package clientkit

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Credential reads an owner-managed key, creating one only when requested.
// Existing files are never overwritten or followed through symlinks.
func Credential(path string, create bool) (string, error) {
	info, err := os.Lstat(path)
	if os.IsNotExist(err) && create {
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			return "", err
		}
		b := make([]byte, 32)
		if _, err := rand.Read(b); err != nil {
			return "", err
		}
		key := hex.EncodeToString(b)
		f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
		if err != nil {
			return "", err
		}
		data, encodeErr := encodePrivate([]byte(key + "\n"))
		if encodeErr != nil {
			f.Close()
			os.Remove(path)
			return "", encodeErr
		}
		_, err = f.Write(data)
		ce := f.Close()
		if err != nil {
			os.Remove(path)
			return "", err
		}
		if ce != nil {
			os.Remove(path)
			return "", ce
		}
		return key, nil
	}
	if err != nil {
		return "", err
	}
	if !info.Mode().IsRegular() {
		return "", fmt.Errorf("credential must be a regular file")
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	b, err = decodePrivate(b, info)
	if err != nil {
		return "", err
	}
	key := strings.TrimSpace(string(b))
	if len(key) < 32 || strings.ContainsAny(key, "\r\n\x00") {
		return "", fmt.Errorf("credential must contain at least 32 characters on one line")
	}
	return key, nil
}

// WritePrivate creates a private connection file without following an existing
// link or overwriting an existing profile. Windows encrypts it with user DPAPI.
func WritePrivate(path string, data []byte) error {
	encoded, err := encodePrivate(data)
	if err != nil {
		return err
	}
	if err = os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	_, err = f.Write(encoded)
	closeErr := f.Close()
	if err == nil {
		err = closeErr
	}
	if err != nil {
		os.Remove(path)
	}
	return err
}

func ReadPrivate(path string) ([]byte, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("private file must be a regular file")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return decodePrivate(data, info)
}

// ReplacePrivate stages an encrypted/private file beside its destination and
// atomically replaces it. A crashed previous save cannot block the next save.
func ReplacePrivate(path string, data []byte) error {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		return err
	}
	tmp := path + "." + hex.EncodeToString(b) + ".tmp"
	if err := WritePrivate(tmp, data); err != nil {
		return err
	}
	defer os.Remove(tmp)
	return replacePrivateFile(tmp, path)
}
