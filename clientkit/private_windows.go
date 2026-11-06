package clientkit

import (
	"bytes"
	"fmt"
	"os"
	"unsafe"

	"golang.org/x/sys/windows"
)

const privatePrefix = "ggrun-dpapi-v1\n"

func replacePrivateFile(from, to string) error {
	src, err := windows.UTF16PtrFromString(from)
	if err != nil {
		return err
	}
	dst, err := windows.UTF16PtrFromString(to)
	if err != nil {
		return err
	}
	return windows.MoveFileEx(src, dst, windows.MOVEFILE_REPLACE_EXISTING|windows.MOVEFILE_WRITE_THROUGH)
}

// DPAPI uses the current Windows user's credential, never machine-wide scope.
func privateCrypt(data []byte, encrypt bool) ([]byte, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("empty private data")
	}
	in := windows.DataBlob{Size: uint32(len(data)), Data: &data[0]}
	var out windows.DataBlob
	var err error
	if encrypt {
		err = windows.CryptProtectData(&in, nil, nil, 0, nil, windows.CRYPTPROTECT_UI_FORBIDDEN, &out)
	} else {
		err = windows.CryptUnprotectData(&in, nil, nil, 0, nil, windows.CRYPTPROTECT_UI_FORBIDDEN, &out)
	}
	if err != nil {
		return nil, fmt.Errorf("Windows credential protection: %w", err)
	}
	defer windows.LocalFree(windows.Handle(unsafe.Pointer(out.Data)))
	return append([]byte(nil), unsafe.Slice(out.Data, int(out.Size))...), nil
}
func encodePrivate(data []byte) ([]byte, error) {
	b, err := privateCrypt(data, true)
	if err != nil {
		return nil, err
	}
	return append([]byte(privatePrefix), b...), nil
}
func decodePrivate(data []byte, _ os.FileInfo) ([]byte, error) {
	if !bytes.HasPrefix(data, []byte(privatePrefix)) {
		return nil, fmt.Errorf("credential is not protected for this Windows user; reconnect or generate a new gateway credential")
	}
	return privateCrypt(data[len(privatePrefix):], false)
}
