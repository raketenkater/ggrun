package clientkit

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestCredentialStablePrivateAndNoSymlink(t *testing.T) {
	path := filepath.Join(t.TempDir(), "key")
	key, err := Credential(path, true)
	if err != nil {
		t.Fatal(err)
	}
	again, err := Credential(path, true)
	if err != nil || again != key {
		t.Fatal("credential changed", err)
	}
	link := path + "-link"
	if err = os.Symlink(path, link); err == nil {
		if _, err = Credential(link, false); err == nil {
			t.Fatal("followed symlink")
		}
	}
	if runtime.GOOS != "windows" {
		if err = os.Chmod(path, 0644); err != nil {
			t.Fatal(err)
		}
		if _, err = Credential(path, false); err == nil {
			t.Fatal("accepted public credential")
		}
	}
}
func TestPrivateReplacementSurvivesPreviousSave(t *testing.T) {
	path := filepath.Join(t.TempDir(), "profile")
	if err := WritePrivate(path, []byte("first")); err != nil {
		t.Fatal(err)
	}
	if err := WritePrivate(path, []byte("overwrite")); err == nil {
		t.Fatal("create overwrote existing profile")
	}
	if err := ReplacePrivate(path, []byte("next")); err != nil {
		t.Fatal(err)
	}
	b, err := ReadPrivate(path)
	if err != nil || string(b) != "next" {
		t.Fatal(string(b), err)
	}
}
