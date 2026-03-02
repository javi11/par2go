package gf16

import (
	"fmt"
	"testing"
)

func TestFastPathDetect(t *testing.T) {
	fmt.Println("NeedsPrepare:", NeedsPrepare())
	fmt.Println("HasMultiPacked:", HasMultiPacked())
	fmt.Println("IdealInputMultiple:", IdealInputMultiple())
}
