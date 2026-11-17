#!/usr/bin/env python3
"""Drive the inspector UI in a real browser and check what it renders.

Runs against an already-running ``main.py ui`` server. Every check exercises a
control the way a person would: clicking layers, hovering a match, switching
tabs, choosing a reference, running an occlusion sweep. Console errors and
failed requests fail the run, because a broken control often looks fine until
the browser complains.

Usage:
    make ui ARGS="--models sift roma --port 8815"
    uv run --with playwright python scripts/ui_smoke.py --url http://127.0.0.1:8815
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, sync_playwright

__all__ = ["main"]

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    """Record one assertion.

    Args:
        name: Check name.
        ok: Whether it passed.
        detail: Optional extra information.
    """
    CHECKS.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}{f' — {detail}' if detail else ''}")


@contextmanager
def browser_page(url: str, screenshots: Path, name: str) -> Iterator[Page]:
    """Open the UI and collect page errors.

    Args:
        url: Inspector base URL.
        screenshots: Directory for screenshots.
        name: Screenshot prefix.

    Yields:
        The page under test.
    """
    errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.on(
            "console",
            lambda message: (
                errors.append(message.text) if message.type == "error" else None
            ),
        )
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url, wait_until="networkidle")
        try:
            yield page
        finally:
            check("no console errors", not errors, "; ".join(errors[:3]))
            page.screenshot(path=str(screenshots / f"{name}.png"))
            browser.close()


def wait_pairs(page: Page) -> None:
    """Wait until the pair list is populated.

    Args:
        page: Page under test.
    """
    page.wait_for_selector("#pair-list li", timeout=60_000)


def inspect_view(page: Page) -> None:
    """Exercise the inspector controls.

    Args:
        page: Page under test.
    """
    wait_pairs(page)
    count = page.locator("#pair-list li").count()
    check("pair list populated", count > 0, f"{count} rows")

    page.wait_for_selector("text.matches, #viewer svg circle", timeout=60_000)
    circles = page.locator("#viewer svg circle").count()
    check("correspondences drawn", circles > 0, f"{circles} points")

    labels = page.locator("#viewer svg text").count()
    check("reference labels drawn", labels > 0, f"{labels} labels")

    meta = page.inner_text("#meta")
    check("meta bar reports matches", "matches" in meta.lower(), meta[:60])

    page.locator("label.chip:has(#labels)").click()
    page.wait_for_timeout(300)
    check(
        "labels toggle hides labels",
        page.locator("#viewer svg text").count() == 0,
    )
    page.locator("label.chip:has(#labels)").click()
    page.wait_for_timeout(300)
    check(
        "labels toggle restores labels",
        page.locator("#viewer svg text").count() > 0,
    )

    first = page.locator("#viewer svg circle").first
    first.hover()
    page.wait_for_timeout(200)
    tooltip_visible = page.locator("#tooltip").is_visible()
    tooltip_text = page.inner_text("#tooltip") if tooltip_visible else ""
    check(
        "hover shows tooltip",
        tooltip_visible and "#" in tooltip_text,
        tooltip_text[:40],
    )

    first.click(force=True)
    page.wait_for_timeout(200)
    check("click pins the match", "pinned" in page.inner_text("#meta"))

    before = page.locator("#viewer svg g").first.get_attribute("transform")
    page.mouse.move(900, 500)
    page.mouse.wheel(0, -240)
    page.wait_for_timeout(200)
    after = page.locator("#viewer svg g").first.get_attribute("transform")
    check("wheel zooms", before != after, f"{before} -> {after}")

    page.select_option("#model", "roma")
    page.wait_for_function(
        "() => document.querySelector('#meta').innerText.includes('roma')",
        timeout=120_000,
    )
    check("model switch reloads the dump", "roma" in page.inner_text("#meta"))


def references_view(page: Page, screenshots: Path, *, allow_writes: bool) -> None:
    """Exercise the reference picker.

    Saving is opt-in because it writes the real selection file.

    Args:
        page: Page under test.
        screenshots: Screenshot directory.
        allow_writes: Whether to click Save and restore the file afterwards.
    """
    selection = Path("configs/references.json")
    backup = selection.read_text() if selection.exists() else None
    page.click("#tab-references")
    page.wait_for_selector("#cand-grid .cand", timeout=120_000)
    candidates = page.locator("#cand-grid .cand").count()
    check("candidates listed", candidates > 1, f"{candidates} crops")

    page.locator("#cand-grid .cand").nth(1).click()
    page.wait_for_timeout(200)
    active = page.locator("#cand-grid .cand.active").count()
    check("clicking a crop selects it", active == 1)

    if not allow_writes:
        check("save skipped", True, "use --allow-writes to exercise it")
        return
    page.click("#ref-save")
    page.wait_for_timeout(800)
    saved = page.inner_text("#ref-saved")
    check("save reports success", "saved" in saved.lower(), saved)
    if backup is not None:
        selection.write_text(backup)
    elif selection.exists():
        selection.unlink()


def ablations_view(page: Page) -> None:
    """Exercise the ablation tab.

    Args:
        page: Page under test.
    """
    page.click("#tab-ablations")
    page.wait_for_selector("#run-table tbody tr", timeout=60_000)
    rows = page.locator("#run-table tbody tr").count()
    check("runs table populated", rows > 0, f"{rows} rows")
    baseline = page.locator("#run-table tbody tr.baseline").count()
    check("baseline row marked", baseline == 1)


def help_panel(page: Page) -> None:
    """Open the help popover.

    Args:
        page: Page under test.
    """
    page.click("#help-button")
    page.wait_for_timeout(200)
    visible = page.locator(".help-panel").is_visible()
    text = page.inner_text(".help-panel") if visible else ""
    check("help panel opens", visible and "Model" in text)
    page.keyboard.press("Escape")


def dataset_switch(page: Page) -> None:
    """Switch datasets and confirm the single-reference message.

    Args:
        page: Page under test.
    """
    page.select_option("#dataset", "rpc")
    page.wait_for_function(
        "() => document.querySelector('#counts').innerText.includes('1244')",
        timeout=60_000,
    )
    check("dataset switch reloads pairs", True)
    page.click("#tab-references")
    page.wait_for_timeout(1500)
    single = page.locator(".references.single").count()
    check("single-reference dataset hides the picker", single == 1)


def occlusion(page: Page) -> None:
    """Run one occlusion sweep and wait for it to finish.

    Args:
        page: Page under test.
    """
    page.select_option("#dataset", "flickrlogos32")
    page.click("#tab-inspector")
    wait_pairs(page)
    page.click("#occlusion")
    page.wait_for_function(
        "() => !document.querySelector('#occlusion').disabled",
        timeout=180_000,
    )
    failed = "failed" in page.inner_text("#meta").lower()
    check("occlusion sweep completes", not failed, page.inner_text("#meta")[:60])


def main(argv: list[str] | None = None) -> int:
    """Run the UI smoke checks.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Smoke-test the inspector UI.")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument(
        "--screenshots",
        type=Path,
        default=Path("outputs/ui/smoke"),
    )
    parser.add_argument("--skip-occlusion", action="store_true")
    parser.add_argument(
        "--allow-writes",
        action="store_true",
        help="Exercise Save, restoring configs/references.json afterwards.",
    )
    args = parser.parse_args(argv)
    args.screenshots.mkdir(parents=True, exist_ok=True)

    try:
        with browser_page(
            f"{args.url}/?model=sift#inspector", args.screenshots, "inspector"
        ) as page:
            inspect_view(page)
        with browser_page(
            f"{args.url}/#references", args.screenshots, "references"
        ) as page:
            references_view(page, args.screenshots, allow_writes=args.allow_writes)
        with browser_page(
            f"{args.url}/#ablations", args.screenshots, "ablations"
        ) as page:
            ablations_view(page)
        with browser_page(
            f"{args.url}/?help=1#inspector", args.screenshots, "help"
        ) as page:
            help_panel(page)
        with browser_page(
            f"{args.url}/#inspector", args.screenshots, "datasets"
        ) as page:
            dataset_switch(page)
        if not args.skip_occlusion:
            with browser_page(
                f"{args.url}/?model=sift#inspector", args.screenshots, "occlusion"
            ) as page:
                occlusion(page)
    except PlaywrightError as error:
        check("page reached", False, str(error))

    failed = [item for item in CHECKS if not item[1]]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("failed: " + ", ".join(name for name, _, _ in failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
