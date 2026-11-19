#!/usr/bin/env python3
"""Conversational, explainable resume-matching agent.

The agent uses the MCP filesystem server for every resume discovery/read. Its
workflow is a LangGraph state machine when ``langgraph`` is installed; the
compatibility runner keeps this classroom project runnable with the standard
library too.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypedDict

try:
    from langgraph.graph import END, START, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:  # dependency-free demo/test fallback
    LANGGRAPH_AVAILABLE = False
    START, END = "__start__", "__end__"


class McpClient:
    """Minimal synchronous stdio JSON-RPC MCP client for one server process."""
    def __init__(self, root: str, server_path: Path | None = None, trace: bool = False) -> None:
        server_path = server_path or Path(__file__).with_name("filesystem_mcp_server.py")
        self.process = subprocess.Popen([sys.executable, str(server_path), "--root", root], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.request_id = 0
        self.trace = trace
        self.request("initialize", {"protocolVersion": "2024-11-05", "clientInfo": {"name": "resume-matcher", "version": "1.0.0"}, "capabilities": {}})
        self.notify("notifications/initialized")

    def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self.request_id += 1
        assert self.process.stdin and self.process.stdout
        request = {"jsonrpc": "2.0", "id": self.request_id, "method": method, "params": params or {}}
        if self.trace: print(f"MCP -> {json.dumps(request)}", file=sys.stderr)
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        response = json.loads(self.process.stdout.readline())
        if self.trace: print(f"MCP <- {json.dumps(response)}", file=sys.stderr)
        if "error" in response:
            raise RuntimeError(f"MCP {response['error']['code']}: {response['error']['message']}")
        return response["result"]

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        assert self.process.stdin
        self.process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method, "params": params or {}}) + "\n")
        self.process.stdin.flush()

    def tool(self, name: str, **arguments: Any) -> Any:
        return self.request("tools/call", {"name": name, "arguments": arguments})["structuredContent"]

    def close(self) -> None:
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=2)
        if self.process.stdout and not self.process.stdout.closed:
            self.process.stdout.close()

    def __enter__(self) -> "McpClient": return self
    def __exit__(self, *_: object) -> None: self.close()


class AgentState(TypedDict, total=False):
    """Persistent graph state: history, understanding, shortlist and rationale."""
    messages: list[dict[str, str]]
    job_description: str
    requirements: dict[str, Any]
    candidates: dict[str, dict[str, Any]]
    retrieved_ids: list[str]
    shortlist: list[dict[str, Any]]
    reasoning: dict[str, str]
    screening: dict[str, list[dict[str, Any]]]
    recommendations: dict[str, str]
    report: str
    feedback: str


SKILLS = ("Python", "Java", "JavaScript", "TypeScript", "React", "Angular", "Vue", "Node.js", "SQL", "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Linux", "CI/CD", "Machine Learning", "Pandas", "PyTorch", "NLP", "LLMs", "Figma", "Salesforce", "Product Management", "Microservices")
STOP_WORDS = {"and", "the", "for", "with", "from", "that", "this", "are", "you", "your", "years", "experience", "have", "will", "our", "all", "candidate", "candidates", "find", "me"}


def keywords(text: str) -> set[str]:
    return {word.lower() for word in re.findall(r"[A-Za-z][A-Za-z+#./-]{1,}", text) if word.lower() not in STOP_WORDS}


def _skills_in(text: str) -> list[str]:
    return [skill for skill in SKILLS if re.search(r"(?<!\w)" + re.escape(skill) + r"(?!\w)", text, re.I)]


def extract_requirements(jd: str) -> dict[str, Any]:
    """Required tool: parse a JD into must-have/nice-to-have criteria."""
    years = re.search(r"(\d+)\s*\+?\s*years?", jd, re.I)
    all_skills = _skills_in(jd)
    nice = []
    for skill in all_skills:
        hit = re.search(re.escape(skill), jd, re.I)
        context = jd[max(0, hit.start() - 20):hit.end() + 35] if hit else ""
        if re.search(r"nice[- ]to[- ]have|preferred|bonus|\bplus\b", context, re.I):
            nice.append(skill)
    must = [skill for skill in all_skills if skill not in nice]
    return {"must_have": must or sorted(keywords(jd)), "nice_to_have": nice,
            "min_years": int(years.group(1)) if years else 0, "source": jd}


def _parse_candidate(path: str, content: str) -> dict[str, Any]:
    name, title = re.search(r"^Name:\s*(.+)$", content, re.M), re.search(r"^Title:\s*(.+)$", content, re.M)
    years = [int(value) for value in re.findall(r"(\d+)\+?\s+years", content, re.I)]
    return {"id": Path(path).stem, "resume": path, "name": name.group(1).strip() if name else Path(path).stem,
            "title": title.group(1).strip() if title else "Not specified", "skills": _skills_in(content),
            "years": max(years, default=0), "content": content}


def _score(candidate: dict[str, Any], req: dict[str, Any]) -> tuple[float, list[str], list[str]]:
    known = {skill.lower() for skill in candidate["skills"]}
    must, nice = req["must_have"], req["nice_to_have"]
    matched_must = [skill for skill in must if skill.lower() in known]
    matched_nice = [skill for skill in nice if skill.lower() in known]
    score = 55 * len(matched_must) / max(1, len(must))
    score += 15 * len(matched_nice) / max(1, len(nice)) if nice else 15
    score += 20 if not req["min_years"] else 20 * min(candidate["years"] / req["min_years"], 1)
    score += 10 if any(term in candidate["title"].lower() for term in keywords(req["source"])) else 4
    return round(min(100, score), 1), matched_must + matched_nice, [skill for skill in must if skill not in matched_must]


@dataclass
class _CompatibilityGraph:
    """Same declared node order if LangGraph has not yet been installed."""
    nodes: list[Callable[[AgentState], dict[str, Any]]]
    def invoke(self, state: AgentState) -> AgentState:
        result = dict(state)
        for node in self.nodes: result.update(node(result))
        return result


class ResumeMatchingAgent:
    """LangGraph workflow: parse → retrieve → rank → screen → recommend → explain."""
    def __init__(self, client: McpClient) -> None:
        self.client, self.last_state = client, {"messages": [], "shortlist": [], "reasoning": {}}
        self.graph = self._build_graph()

    def parse_jd(self, state: AgentState) -> dict[str, Any]:
        return {"messages": state.get("messages", []) + [{"role": "user", "content": state["job_description"]}]}

    def extract_requirements_node(self, state: AgentState) -> dict[str, Any]:
        return {"requirements": extract_requirements(state["job_description"])}

    def search_resumes(self, state: AgentState) -> dict[str, Any]:
        files = self.client.tool("list_directory", path=".")
        paths = [item["path"] for item in files if item["type"] == "file" and item["path"].lower().endswith((".txt", ".md"))]
        batch = self.client.tool("batch_process", paths=paths)
        parsed = [_parse_candidate(item["path"], item["content"]) for item in batch["processed"]]
        candidates = {item["id"]: item for item in parsed}
        retrieved = self.rag_search(candidates, state["job_description"])
        return {"candidates": candidates, "retrieved_ids": retrieved}

    def rag_search(self, candidates: dict[str, dict[str, Any]], query: str) -> list[str]:
        """RAG search tool: rank resume documents by lexical query evidence.

        The MCP batch provides the document corpus; this method is deliberately
        isolated so an embedding/vector search from Milestone 2 can replace the
        lexical scorer without changing graph nodes.
        """
        terms = keywords(query)
        return sorted(candidates, key=lambda cid: len(terms & keywords(candidates[cid]["content"])), reverse=True)

    def rank_candidates(self, state: AgentState) -> dict[str, Any]:
        ranked, reasoning = [], {}
        for cid in state["retrieved_ids"]:
            raw = state["candidates"][cid]; score, matched, gaps = _score(raw, state["requirements"])
            item = {key: raw[key] for key in ("id", "name", "title", "resume", "skills", "years")}
            item.update({"score": score, "matched": matched, "gaps": gaps}); ranked.append(item)
            reasoning[cid] = f"{raw['name']} scored {score}/100: matched {', '.join(matched) or 'no required skills'}; {raw['years']} stated years versus {state['requirements']['min_years']} requested; missing {', '.join(gaps) or 'no stated must-have skills'}."
        ranked.sort(key=lambda item: (-item["score"], item["name"]))
        return {"shortlist": ranked[:10], "reasoning": reasoning}

    def deep_analysis(self, state: AgentState) -> dict[str, Any]:
        return {"screening": {item["id"]: [{"round": "initial", "result": "pass" if item["score"] >= 45 else "review"}, {"round": "deep", "result": "pass" if item["score"] >= 60 else "review", "gaps": item["gaps"]}] for item in state["shortlist"]}}

    def final_recommendations(self, state: AgentState) -> dict[str, Any]:
        return {"recommendations": {item["id"]: "HIRE" if item["score"] >= 78 and not item["gaps"] else "HOLD / INTERVIEW" if item["score"] >= 55 else "NO HIRE" for item in state["shortlist"]}}

    def generate_report_node(self, state: AgentState) -> dict[str, Any]:
        req = state["requirements"]
        lines = ["Match report", f"Must-have: {', '.join(req['must_have']) or 'not specified'}", f"Nice-to-have: {', '.join(req['nice_to_have']) or 'none'}", "", "Top 10 initial screen:"]
        for index, item in enumerate(state["shortlist"], 1):
            lines.extend([f"{index}. {item['name']} ({item['id']}) — {item['score']:.1f}/100 — {state['recommendations'][item['id']]}", f"   Strengths: {', '.join(item['matched']) or 'no direct skill evidence'}. Gaps: {', '.join(item['gaps']) or 'none'}." ])
        return {"report": "\n".join(lines)}

    def human_feedback_loop(self, state: AgentState) -> dict[str, Any]:
        feedback = state.get("feedback", "")
        return {"messages": state.get("messages", []) + ([{"role": "assistant", "content": "Feedback recorded: " + feedback}] if feedback else [])}

    def _build_graph(self) -> Any:
        functions = [("parse_jd", self.parse_jd), ("extract_requirements", self.extract_requirements_node), ("search_resumes", self.search_resumes), ("rank_candidates", self.rank_candidates), ("deep_analysis", self.deep_analysis), ("final_recommendations", self.final_recommendations), ("generate_report", self.generate_report_node), ("human_feedback", self.human_feedback_loop)]
        if not LANGGRAPH_AVAILABLE: return _CompatibilityGraph([function for _, function in functions])
        graph = StateGraph(AgentState)
        for name, function in functions: graph.add_node(name, function)
        graph.add_edge(START, "parse_jd")
        for (left, _), (right, _) in zip(functions, functions[1:]): graph.add_edge(left, right)
        graph.add_edge("human_feedback", END)
        return graph.compile()

    def _candidate(self, candidate_id: str) -> dict[str, Any] | None:
        needle = candidate_id.lower()
        return next((item for item in self.last_state.get("shortlist", []) if item["id"].lower() == needle or item["name"].lower() == needle), None)

    def compare_candidates(self, candidate_ids: list[str]) -> str:
        """Required tool: a head-to-head comparison of candidates in the shortlist."""
        wanted = {item.lower() for item in candidate_ids}; matches = [item for item in self.last_state.get("shortlist", []) if item["id"].lower() in wanted]
        if not matches: return "No shortlisted candidates matched those IDs. Run a search first."
        return "Candidate comparison\n" + "\n".join(f"- {item['name']} ({item['id']}): {item['score']}/100; skills: {', '.join(item['matched']) or 'none'}; gaps: {', '.join(item['gaps']) or 'none'}." for item in matches)

    def generate_interview_questions(self, candidate_id: str) -> list[str]:
        """Required tool: focused screening questions based on evidence and gaps."""
        candidate = self._candidate(candidate_id)
        if not candidate: return ["Candidate not found. Run a search first."]
        strength, gap = (candidate["matched"] or ["core experience"])[0], (candidate["gaps"] or ["recent project depth"])[0]
        return [f"Walk me through a production project where you used {strength}.", f"How would you close the gap in {gap} during your first 90 days?", "Describe a difficult technical trade-off you made and how you measured the result."]

    def run_search(self, job_description: str, feedback: str = "") -> AgentState:
        self.last_state = self.graph.invoke({"job_description": job_description, "feedback": feedback, "messages": self.last_state.get("messages", [])})
        return self.last_state

    def answer(self, query: str) -> str:
        """Natural-language chat router for search, comparison, explanation, and refinement."""
        if re.search(r"\b(compare|side by side)\b", query, re.I):
            ids = re.findall(r"\bR?\d{1,3}\b", query, re.I)
            return self.compare_candidates(ids or [item["id"] for item in self.last_state.get("shortlist", [])[:3]])
        why = re.search(r"why\s+(?:did\s+)?(.+?)(?:\s+rank|\?|$)", query, re.I)
        if why:
            candidate = self._candidate(why.group(1).strip())
            return self.last_state.get("reasoning", {}).get(candidate["id"], "Candidate not found.") if candidate else "Candidate not found. Use an ID such as R001."
        question = re.search(r"(?:questions? (?:for|about)|screen)\s+([\w-]+)", query, re.I)
        if question: return "\n".join("- " + item for item in self.generate_interview_questions(question.group(1)))
        refine = bool(self.last_state.get("shortlist")) and bool(re.search(r"\b(add|require|must|only|instead|change|update|at least)\b", query, re.I))
        state = self.run_search(query if not refine else self.last_state.get("job_description", "") + "; " + query, query if refine else "")
        return state["report"] + ("\nRefined using your latest criteria; rankings were recalculated." if refine else "")

    def match(self, job_description: str, directory: str = ".") -> list[dict[str, Any]]:
        """Backwards-compatible programmatic ranking API; files remain MCP-managed."""
        del directory
        return [{"resume": item["resume"], "score": item["score"], "matched_keywords": item["matched"]} for item in self.run_search(job_description)["shortlist"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph conversational resume matching agent")
    parser.add_argument("job_description", nargs="?")
    parser.add_argument("--root", default="resumes", help="MCP resume directory")
    parser.add_argument("--chat", action="store_true", help="Start an interactive natural-language chat")
    parser.add_argument("--trace", action="store_true", help="Print MCP JSON-RPC requests and responses to stderr")
    args = parser.parse_args()
    with McpClient(args.root, trace=args.trace) as client:
        agent = ResumeMatchingAgent(client)
        if args.chat:
            print("Resume agent ready. Ask for candidates, comparisons, explanations, or screening questions. Type 'quit' to exit.")
            while True:
                try:
                    query = input("you> ").strip()
                except EOFError:
                    break
                if query.lower() in {"quit", "exit"}:
                    break
                if query:
                    print(agent.answer(query))
        elif args.job_description:
            print(agent.answer(args.job_description))
        else:
            parser.error("provide a job description or --chat")


if __name__ == "__main__":
    main()
