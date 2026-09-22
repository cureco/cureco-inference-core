"""Check Trusted Publishing token exchange without uploading a distribution."""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def read_json(request):
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main():
    registries = {"pypi": "https://pypi.org", "testpypi": "https://test.pypi.org"}
    if len(sys.argv) != 2 or sys.argv[1] not in registries:
        raise SystemExit("Expected pypi or testpypi")
    registry = registries[sys.argv[1]]
    audience = read_json(registry + "/_/oidc/audience")["audience"]
    request_url = os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"]
    separator = "&" if "?" in request_url else "?"
    request = urllib.request.Request(
        request_url + separator + urllib.parse.urlencode({"audience": audience}),
        headers={"Authorization": "Bearer " + os.environ["ACTIONS_ID_TOKEN_REQUEST_TOKEN"]},
    )
    identity = read_json(request)["value"]
    print(f"::add-mask::{identity}")
    request = urllib.request.Request(
        registry + "/_/oidc/mint-token",
        data=json.dumps({"token": identity}).encode(),
        headers={"Content-Type": "application/json"},
    )
    token = read_json(request).get("token")
    if not token:
        raise SystemExit("Registry did not issue a publishing token")
    print(f"::add-mask::{token}")
    print(f"PASS: {sys.argv[1]} Trusted Publishing authentication; no files uploaded")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as error:
        raise SystemExit(f"Trusted Publishing authentication failed: HTTP {error.code}") from None
