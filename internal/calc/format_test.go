package calc

import "testing"

// Expected strings are what the Space's own code prints under node: its
// toFixed, Math.round, fmtGB and fmtParams.

func TestJSToFixedMatchesJavaScript(t *testing.T) {
	cases := []struct {
		x      float64
		digits int
		want   string
	}{
		{0.25, 1, "0.3"},
		{0.35, 1, "0.3"},
		{0.05, 1, "0.1"},
		{0.15, 1, "0.1"},
		{9.95, 1, "9.9"},
		{9.96, 1, "10.0"},
		{0, 1, "0.0"},
		{0.5, 1, "0.5"},
		{1.45, 1, "1.4"},
		{2.25, 1, "2.3"},
		{123.456, 1, "123.5"},
		{9.04, 1, "9.0"},
		{0.0625, 1, "0.1"},
		{3.75, 1, "3.8"},
		{0.45, 1, "0.5"},
		{1.05, 1, "1.1"},
		{9.25, 1, "9.3"},
		{0.5, 0, "1"},
		{1.5, 0, "2"},
		{2.5, 0, "3"},
		{3.5, 0, "4"},
		{0, 0, "0"},
		{102.290824, 0, "102"},
		{751.632384, 0, "752"},
		{0.4, 0, "0"},
		{999.5, 0, "1000"},
		{0.49, 0, "0"},
	}
	for _, c := range cases {
		if got := JSToFixed(c.x, c.digits); got != c.want {
			t.Errorf("JSToFixed(%v, %d) = %q, want %q", c.x, c.digits, got, c.want)
		}
	}
}

func TestFormatGiBMatchesSpace(t *testing.T) {
	cases := []struct {
		x    float64
		want string
	}{
		{9.96, "10.0 GiB"},
		{9.94, "9.9 GiB"},
		{10.5, "11 GiB"},
		{1234.5, "1235 GiB"},
		{2446.182725504, "2446 GiB"},
	}
	for _, c := range cases {
		if got := FormatGiB(c.x); got != c.want {
			t.Errorf("FormatGiB(%v) = %q, want %q", c.x, got, c.want)
		}
	}
}

func TestFormatParamsMatchesSpace(t *testing.T) {
	cases := []struct {
		params float64
		want   string
	}{
		{8190735360, "8.2B"},
		{102290824, "102M"},
		{751632384, "752M"},
		{2446182725504, "2446.2B"},
		{999999999, "1000M"},
		{1e9, "1.0B"},
	}
	for _, c := range cases {
		if got := FormatParams(c.params); got != c.want {
			t.Errorf("FormatParams(%v) = %q, want %q", c.params, got, c.want)
		}
	}
}
