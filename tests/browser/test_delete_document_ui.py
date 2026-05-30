"""Browser E2E tests for the F0 delete-document UI flow.

Playwright intercepts API calls so the mocked uvicorn server only needs to
serve the static HTML/JS/CSS; no real database or Redis is involved.
"""
import pytest
from playwright.sync_api import Page, Request, Route, expect

_FAKE_DOC_ID = "550e8400-e29b-41d4-a716-446655440000"
_FAKE_FILE = {
    "name": "report.pdf",
    "mimeType": "application/pdf",
    "buffer": b"%PDF-1.4 minimal",
}


def _make_api_router(delete_status: int = 204):
    """Return a Playwright route handler that fakes the KB API."""

    def _handler(route: Route, request: Request) -> None:
        url = request.url
        method = request.method
        if "/knowledge-base/documents" not in url:
            route.continue_()
            return
        if url.endswith("/status"):
            route.fulfill(status=200, json={"status": "indexed"})
        elif method == "DELETE":
            route.fulfill(status=delete_status)
        elif method == "POST":
            route.fulfill(status=202, json={"doc_id": _FAKE_DOC_ID})
        else:
            route.continue_()

    return _handler


def _upload_file(page: Page, live_server: str) -> None:
    page.goto(live_server)
    with page.expect_response(
        lambda r: "/knowledge-base/documents" in r.url and r.status == 202
    ):
        page.locator("#file-input").set_input_files(_FAKE_FILE)


class TestDeleteDocumentUI:
    def test_delete_button_visible_after_upload(self, page: Page, live_server: str):
        page.route("**/knowledge-base/**", _make_api_router())
        _upload_file(page, live_server)

        file_item = page.locator(".file-item").first
        expect(file_item).to_be_visible()
        expect(file_item.locator(".delete-btn")).to_be_visible()
        expect(file_item.locator(".file-name")).to_contain_text("report.pdf")

    def test_delete_removes_item_on_204(self, page: Page, live_server: str):
        page.route("**/knowledge-base/**", _make_api_router(delete_status=204))
        _upload_file(page, live_server)

        page.locator(".file-item").first.locator(".delete-btn").click()

        expect(page.locator(".file-item")).to_have_count(0)

    def test_delete_shows_toast_and_reenables_button_on_error(
        self, page: Page, live_server: str
    ):
        page.route("**/knowledge-base/**", _make_api_router(delete_status=500))
        _upload_file(page, live_server)

        delete_btn = page.locator(".file-item").first.locator(".delete-btn")
        delete_btn.click()

        # Toast should appear with the error status
        expect(page.locator(".toast.show")).to_be_visible()
        expect(page.locator(".toast.show")).to_contain_text("500")

        # Item must still be in the list
        expect(page.locator(".file-item")).to_have_count(1)

        # Button re-enabled so the user can retry
        expect(delete_btn).to_be_enabled()
