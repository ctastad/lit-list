#!/usr/bin/env python3
"""
list_to_prompt.py — copy every book from a Hardcover list into a prompt.

  export HARDCOVER_TOKEN="..."     # read:lists + write:prompts

  python list_to_prompt.py --find                     # discover list + prompt IDs
  python list_to_prompt.py --find --match club

  python list_to_prompt.py --list-id 1234 --prompt-id 5678            # dry run
  python list_to_prompt.py --list-id 1234 --prompt-id 5678 --apply

  python list_to_prompt.py --list-id 1234 --create "Which next?" \
      --description "Our shortlist" --apply
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

ENDPOINT = "https://api.hardcover.app/v1/graphql"
UA = "hardcover-list-to-prompt/1.1 (book club)"
PAGE = 1000

ME_Q = "query { me { id username } }"

LISTS_Q = """
query L($uid: Int!) {
  lists(where: {user_id: {_eq: $uid}}, order_by: {updated_at: desc}, limit: 100) {
    id name slug books_count ranked privacy_setting_id
  }
}
"""

PROMPTS_Q = """
query P($uid: Int!) {
  prompts(where: {user_id: {_eq: $uid}}, order_by: {id: desc}, limit: 100) {
    id question answers_count books_count
  }
}
"""

LIST_Q = """
query L($id: Int!) {
  lists(where: {id: {_eq: $id}}, limit: 1) {
    id name list_books(limit: 1000) { position book_id book { id title } }
  }
}
"""

EXISTING_Q = """
query E($pid: Int!, $limit: Int!, $offset: Int!) {
  prompt_answers(where: {prompt_id: {_eq: $pid}}, limit: $limit, offset: $offset) {
    id book_id user_id
  }
}
"""

PROMPT_INFO_Q = """
query I($id: Int!) {
  prompts_by_pk(id: $id) { id question privacy_setting_id answers_count books_count }
}
"""

INSERT_PROMPT = """
mutation P($o: CreatePromptInput!) {
  insert_prompt(object: $o) { id error prompt { id question } }
}
"""

INSERT_ANSWER = """
mutation A($o: PromptAnswerCreateInput!) {
  insert_prompt_answer(object: $o) { id book_id prompt_id user_id }
}
"""


def gql(query, variables, token, retries=5):
    body = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}",
                 "User-Agent": UA})
    for _ in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                payload = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            if e.code == 429:
                wait = int(e.headers.get("Retry-After", 10))
                print(f"  rate limited; sleeping {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            raise SystemExit(f"HTTP {e.code}: {detail}")
        if payload.get("errors"):
            raise RuntimeError(json.dumps(payload["errors"], indent=2))
        return payload["data"]
    raise SystemExit("Gave up after repeated throttling.")


def whoami(token):
    me = gql(ME_Q, {}, token)["me"]
    if not me:
        raise SystemExit("`me` returned nothing — check the token.")
    return me[0]


def find(token, match):
    me = whoami(token)
    print(f"@{me['username']} (user {me['id']})\n")

    lists = gql(LISTS_Q, {"uid": me["id"]}, token)["lists"]
    prompts = gql(PROMPTS_Q, {"uid": me["id"]}, token)["prompts"]
    if match:
        m = match.lower()
        lists = [x for x in lists if m in x["name"].lower()]
        prompts = [x for x in prompts if m in (x["question"] or "").lower()]

    print("LISTS")
    if not lists:
        print("  (none)")
    for x in lists:
        print(f"  {x['id']:>8}  {x['books_count']:>3} books  ranked={str(x['ranked']):<5}  {x['name']}")

    print("\nPROMPTS")
    if not prompts:
        print("  (none)")
    for x in prompts:
        q = (x["question"] or "").replace("\n", " ")
        q = q[:70] + ("…" if len(q) > 70 else "")
        print(f"  {x['id']:>8}  {x['answers_count']:>3} answers  {q}")

    print("\nUse: --list-id <id> --prompt-id <id>")


def fetch_list(token, list_id):
    lists = gql(LIST_Q, {"id": list_id}, token).get("lists") or []
    if not lists:
        raise SystemExit(f"List {list_id} not visible to this token.")
    lst = lists[0]
    return lst, sorted(lst["list_books"],
                       key=lambda b: (b.get("position") is None, b.get("position") or 0))


def fetch_answered(token, prompt_id):
    seen, offset = set(), 0
    while True:
        rows = gql(EXISTING_Q, {"pid": prompt_id, "limit": PAGE, "offset": offset},
                   token)["prompt_answers"]
        seen.update(r["book_id"] for r in rows)
        if len(rows) < PAGE:
            return seen
        offset += PAGE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--find", action="store_true", help="list your lists and prompts, with IDs")
    ap.add_argument("--match", help="substring filter for --find")
    ap.add_argument("--list-id", type=int)
    ap.add_argument("--prompt-id", type=int)
    ap.add_argument("--create", metavar="QUESTION")
    ap.add_argument("--description", default="", help="required by the API when --create")
    ap.add_argument("--privacy", type=int, default=1, help="1=public 2=followers 3=private")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HARDCOVER_TOKEN")
    if not token:
        raise SystemExit("Set HARDCOVER_TOKEN first.")

    if args.find:
        find(token, args.match)
        return

    if not args.list_id or (not args.prompt_id and not args.create):
        raise SystemExit("Need --list-id and either --prompt-id or --create (or use --find).")

    lst, books = fetch_list(token, args.list_id)
    print(f'List "{lst["name"]}": {len(books)} books\n')

    prompt_id = args.prompt_id
    if args.create:
        res = gql(INSERT_PROMPT, {"o": {"question": args.create,
                                        "description": args.description,
                                        "privacy_setting_id": args.privacy}}, token)["insert_prompt"]
        if res.get("error"):
            raise SystemExit(f"Prompt create failed: {res['error']}")
        prompt_id = res["id"]
        print(f'Created prompt {prompt_id}: "{args.create}"\n')

    info = gql(PROMPT_INFO_Q, {"id": prompt_id}, token)["prompts_by_pk"]
    if not info:
        raise SystemExit(f"Prompt {prompt_id} not visible to this token.")
    print(f'Prompt {prompt_id}: "{info["question"]}"')
    print(f'  privacy={info["privacy_setting_id"]}  '
          f'answers={info["answers_count"]}  books={info["books_count"]}')
    if info["privacy_setting_id"] == 3:
        print("  ! private — members can't see or answer it")
    print()

    done = fetch_answered(token, prompt_id)
    todo = [b for b in books if b["book_id"] not in done]
    skip = len(books) - len(todo)
    for b in todo:
        print(f"  + {(b.get('book') or {}).get('title') or b['book_id']}")
    if skip:
        print(f"\n  ({skip} already answered — skipped)")
    if not todo:
        print("\nNothing to do.")
        return
    if not args.apply:
        print(f"\nDry run: would add {len(todo)} answer(s). Add --apply.")
        return

    ok = failed = 0
    for i, b in enumerate(todo, 1):
        title = (b.get("book") or {}).get("title") or str(b["book_id"])
        try:
            r = gql(INSERT_ANSWER,
                    {"o": {"prompt_id": prompt_id, "book_id": b["book_id"]}}, token
                    )["insert_prompt_answer"]
            if i <= 3 or i % 25 == 0:
                print(f"  {title} -> answer {r['id']} (user {r.get('user_id')})")
            ok += 1
        except (RuntimeError, SystemExit) as e:
            print(f"  ! {title}: {e}", file=sys.stderr)
            failed += 1

    after = gql(PROMPT_INFO_Q, {"id": prompt_id}, token)["prompts_by_pk"]
    print(f"\nAdded {ok}, failed {failed}.")
    print(f'Prompt reports answers={after["answers_count"]}, books={after["books_count"]}')


if __name__ == "__main__":
    main()
