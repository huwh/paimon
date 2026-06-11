#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build and send the weekday Apache Paimon community digest.

The digest intentionally uses only public sources and Python's standard library:
* GitHub REST API for apache/paimon commits, pull requests, and issues.
* lists.apache.org Pony Mail API for Paimon/Flink mailing-list threads.
* SMTP for delivery, configured through environment variables.
"""

import argparse
import datetime as dt
import email.message
import json
import os
import re
import smtplib
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

GITHUB_API = "https://api.github.com"
LISTS_API = "https://lists.apache.org/api/stats.lua"
DEFAULT_REPO = "apache/paimon"
DEFAULT_MAIL_LISTS = (
    "dev@paimon.apache.org",
    "user@paimon.apache.org",
    "dev@flink.apache.org",
    "user@flink.apache.org",
)
RELEASE_THREAD_RE = re.compile(
    r"\b(rc\d+|release|vote|voting|candidate|announce)\b", re.IGNORECASE
)
IMPORTANT_MAIL_RE = re.compile(
    r"\b(paimon|flink|pip|fip|design|proposal|discussion|support|bug|feature|connector|cdc|catalog|schema|compaction|snapshot|changelog|stream|batch)\b",
    re.IGNORECASE,
)


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_z(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def request_json(url: str, token: Optional[str] = None, accept: str = "application/vnd.github+json") -> Any:
    headers = {"Accept": accept, "User-Agent": "paimon-community-digest"}
    if token:
        headers["Authorization"] = "Bearer " + token
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def github_get(path: str, token: Optional[str], params: Optional[Dict[str, str]] = None, accept: str = "application/vnd.github+json") -> Any:
    query = "?" + urllib.parse.urlencode(params) if params else ""
    return request_json(GITHUB_API + path + query, token=token, accept=accept)


def github_search(query: str, token: Optional[str], per_page: int = 20) -> List[Dict[str, Any]]:
    data = github_get("/search/issues", token, {"q": query, "sort": "updated", "order": "desc", "per_page": str(per_page)})
    return data.get("items", [])


def split_repo(repo: str) -> Tuple[str, str]:
    owner, name = repo.split("/", 1)
    return owner, name


def load_github_digest(repo: str, since: dt.datetime, until: dt.datetime, token: Optional[str]) -> Dict[str, Any]:
    owner, name = split_repo(repo)
    repo_data = github_get("/repos/%s/%s" % (owner, name), token)
    branch = repo_data.get("default_branch", "master")
    commits = github_get(
        "/repos/%s/%s/commits" % (owner, name),
        token,
        {"sha": branch, "since": iso_z(since), "until": iso_z(until), "per_page": "50"},
    )
    enriched_commits = []
    for commit in commits:
        sha = commit.get("sha", "")
        pulls = []
        try:
            pulls = github_get(
                "/repos/%s/%s/commits/%s/pulls" % (owner, name, sha),
                token,
                {"per_page": "5"},
                accept="application/vnd.github.groot-preview+json",
            )
        except urllib.error.HTTPError:
            pulls = []
        enriched_commits.append({"commit": commit, "pulls": pulls})

    since_day = since.date().isoformat()
    return {
        "branch": branch,
        "commits": enriched_commits,
        "merged_prs": github_search("repo:%s is:pr merged:>=%s" % (repo, since_day), token, per_page=30),
        "opened_prs": github_search("repo:%s is:pr created:>=%s" % (repo, since_day), token, per_page=20),
        "updated_issues": github_search("repo:%s is:issue updated:>=%s" % (repo, since_day), token, per_page=20),
    }


def mail_list_parts(address: str) -> Tuple[str, str]:
    local, domain = address.split("@", 1)
    return local, domain


def load_mail_threads(lists: Sequence[str], since: dt.datetime, until: dt.datetime) -> List[Dict[str, Any]]:
    threads = []
    date_range = "dfr=%s dto=%s" % (since.date().isoformat(), until.date().isoformat())
    for address in lists:
        local, domain = mail_list_parts(address)
        params = {
            "list": local,
            "domain": domain,
            "d": date_range,
            "emailsOnly": "true",
            "q": "paimon OR flink OR PIP OR FIP",
        }
        url = LISTS_API + "?" + urllib.parse.urlencode(params)
        try:
            data = request_json(url, token=None, accept="application/json")
        except Exception as exc:  # keep the digest useful when one archive endpoint is unavailable
            threads.append({"list": address, "error": str(exc), "subject": "mail archive fetch failed"})
            continue
        seen_subjects = set()
        for email_item in data.get("emails", []):
            subject = normalize_subject(email_item.get("subject", ""))
            if not subject or subject in seen_subjects:
                continue
            seen_subjects.add(subject)
            if RELEASE_THREAD_RE.search(subject):
                continue
            body = email_item.get("body", "") or ""
            if not (IMPORTANT_MAIL_RE.search(subject) or IMPORTANT_MAIL_RE.search(body)):
                continue
            threads.append({
                "list": address,
                "subject": subject,
                "from": email_item.get("from", ""),
                "epoch": email_item.get("epoch", 0),
                "url": "https://lists.apache.org/list.html?%s" % address,
                "summary": summarize_text(body or subject, max_words=28),
            })
    threads.sort(key=lambda item: item.get("epoch", 0), reverse=True)
    return threads[:30]


def normalize_subject(subject: str) -> str:
    subject = re.sub(r"^\s*(re|fwd|aw):\s*", "", subject, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", subject).strip()


def summarize_text(value: str, max_words: int = 24) -> str:
    value = re.sub(r"https?://\S+", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    words = value.split()
    if not words:
        return "No text summary available."
    summary = " ".join(words[:max_words])
    return summary + ("..." if len(words) > max_words else "")


def label_names(pull: Dict[str, Any]) -> str:
    labels = [label.get("name", "") for label in pull.get("labels", []) if label.get("name")]
    return ", ".join(labels[:4]) if labels else "unlabeled"


def classify_function(title: str, files: Sequence[str]) -> str:
    text = (title + " " + " ".join(files)).lower()
    if any(word in text for word in ("doc", "readme", "website")):
        return "documentation/community docs"
    if any(word in text for word in ("test", "ci", "build", "maven")):
        return "test/build reliability"
    if "flink" in text:
        return "Flink integration"
    if "spark" in text:
        return "Spark integration"
    if any(word in text for word in ("fix", "bug", "exception", "error")):
        return "bug fix"
    if any(word in text for word in ("support", "add", "introduce", "implement")):
        return "new capability"
    if any(word in text for word in ("improve", "optimize", "refactor")):
        return "improvement/refactor"
    return "core project change"


def format_commit(item: Dict[str, Any]) -> str:
    commit = item["commit"]
    message = commit.get("commit", {}).get("message", "").splitlines()[0]
    sha = commit.get("sha", "")[:8]
    url = commit.get("html_url", "")
    pulls = item.get("pulls") or []
    if pulls:
        pull = pulls[0]
        title = pull.get("title") or message
        files_hint = [pull.get("base", {}).get("ref", "")]
        kind = classify_function(title, files_hint)
        return "- `%s` %s — %s; labels: %s; %s" % (sha, title, kind, label_names(pull), pull.get("html_url", url))
    return "- `%s` %s — %s; %s" % (sha, message, classify_function(message, []), url)


def format_issue(item: Dict[str, Any]) -> str:
    return "- #%s %s — %s" % (item.get("number"), item.get("title"), item.get("html_url"))


def format_mail(item: Dict[str, Any]) -> str:
    if item.get("error"):
        return "- [%s] %s: %s" % (item.get("list"), item.get("subject"), item.get("error"))
    return "- [%s] %s — %s (%s)" % (item.get("list"), item.get("subject"), item.get("summary"), item.get("url"))


def section(title: str, lines: Iterable[str], empty: str) -> str:
    material = list(lines)
    return "## %s\n%s\n" % (title, "\n".join(material) if material else empty)


def build_digest(repo: str, since: dt.datetime, until: dt.datetime, github_data: Dict[str, Any], mail_threads: List[Dict[str, Any]]) -> Tuple[str, str]:
    subject = "Apache Paimon weekday digest: %s" % until.date().isoformat()
    github_warning = ""
    if github_data.get("fetch_error"):
        github_warning = "\n> Warning: GitHub activity fetch failed: %s\n" % github_data["fetch_error"]
    parts = [
        "# Apache Paimon 社区工作日简报\n",
        "Window: %s 至 %s UTC\n" % (iso_z(since), iso_z(until)),
        "Repository: https://github.com/%s (default branch: `%s`)%s\n" % (repo, github_data.get("branch", "master"), github_warning),
        section("最新合入 commits 及功能", (format_commit(item) for item in github_data.get("commits", [])), "- No commits merged in this window."),
        section("社区最新进展（PR / Issue）", (
            ["### Merged PRs"]
            + [format_issue(item) for item in github_data.get("merged_prs", [])[:12]]
            + ["\n### Newly opened PRs"]
            + [format_issue(item) for item in github_data.get("opened_prs", [])[:8]]
            + ["\n### Recently updated issues"]
            + [format_issue(item) for item in github_data.get("updated_issues", [])[:8]]
        ), "- No GitHub community activity found."),
        section("Flink / Paimon 邮件进展（已过滤 release vote 类线程）", (format_mail(item) for item in mail_threads), "- No relevant mailing-list threads found."),
        "\n---\nGenerated by `tools/community_digest.py`.\n",
    ]
    return subject, "\n".join(parts)


def send_email(subject: str, body: str) -> None:
    required = ["DIGEST_SMTP_HOST", "DIGEST_EMAIL_FROM", "DIGEST_EMAIL_TO"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError("Missing email environment variables: " + ", ".join(missing))
    host = os.environ["DIGEST_SMTP_HOST"]
    port = int(os.environ.get("DIGEST_SMTP_PORT") or "587")
    username = os.environ.get("DIGEST_SMTP_USERNAME")
    password = os.environ.get("DIGEST_SMTP_PASSWORD")
    sender = os.environ["DIGEST_EMAIL_FROM"]
    recipients = [part.strip() for part in os.environ["DIGEST_EMAIL_TO"].split(",") if part.strip()]

    message = email.message.EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message.set_content(body)

    context = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(host, port, context=context, timeout=45) as smtp:
            if username and password:
                smtp.login(username, password)
            smtp.send_message(message)
    else:
        with smtplib.SMTP(host, port, timeout=45) as smtp:
            smtp.starttls(context=context)
            if username and password:
                smtp.login(username, password)
            smtp.send_message(message)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("DIGEST_REPO", DEFAULT_REPO))
    parser.add_argument("--lookback-hours", type=int, default=int(os.environ.get("DIGEST_LOOKBACK_HOURS", "24")))
    parser.add_argument("--mail-list", action="append", dest="mail_lists", default=[])
    parser.add_argument("--dry-run", action="store_true", help="Print the digest instead of sending email.")
    parser.add_argument("--skip-fetch", action="store_true", help="Do not call external services; useful for syntax/smoke tests.")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    until = utc_now()
    since = until - dt.timedelta(hours=args.lookback_hours)
    mail_lists = args.mail_lists or list(DEFAULT_MAIL_LISTS)
    if args.skip_fetch:
        github_data = {"branch": "master", "commits": [], "merged_prs": [], "opened_prs": [], "updated_issues": []}
        mail_threads = []
    else:
        token = os.environ.get("GITHUB_TOKEN")
        try:
            github_data = load_github_digest(args.repo, since, until, token)
        except Exception as exc:
            github_data = {
                "branch": "unknown",
                "commits": [],
                "merged_prs": [],
                "opened_prs": [],
                "updated_issues": [],
                "fetch_error": str(exc),
            }
        mail_threads = load_mail_threads(mail_lists, since, until)
    subject, body = build_digest(args.repo, since, until, github_data, mail_threads)
    if args.dry_run:
        print("Subject: " + subject)
        print(body)
    else:
        send_email(subject, body)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
