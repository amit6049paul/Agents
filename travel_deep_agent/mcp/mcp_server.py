"""
mcp/mcp_server.py
------------------
A minimal, local stand-in for a real MCP (Model Context Protocol) server.

A real MCP server exposes tools over JSON-RPC (usually via stdio or SSE) so
ANY compliant client -- Claude Desktop, an IDE, this project -- can discover
and call them the same way. Here we keep the same *shape* of that idea
(register a tool once, discover its schema, call it by name, get a
structured result) but run it in-process so the demo has zero extra moving
parts and no network dependency.

Swapping this for a real MCP connection later only means replacing
`MCPToolServer` with a client from the `mcp` python package that talks to
an external MCP server -- nothing in agents/ or skills/ would need to change,
because they only depend on `list_tools()` and `call_tool()`.
"""
import inspect
import time


class MCPToolServer:
    def __init__(self):
        self._tools = {}          # name -> python function
        self._call_log = []       # simple audit trail, useful for debugging

    def register(self, func) -> None:
        self._tools[func.__name__] = func

    def register_many(self, funcs) -> None:
        for f in funcs:
            self.register(f)

    def list_tools(self, names: list = None) -> list:
        """Return the callable tool objects. If `names` is given, only
        return that subset -- this is how each subagent gets a restricted
        toolbox (e.g. the Budget agent can't book flights)."""
        if names is None:
            return list(self._tools.values())
        return [self._tools[n] for n in names if n in self._tools]

    def call_tool(self, name: str, **kwargs) -> dict:
        """Directly invoke a tool by name (used for manual/scripted calls
        outside of a model's automatic function calling)."""
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        started = time.time()
        result = self._tools[name](**kwargs)
        self._call_log.append({
            "tool": name, "args": kwargs, "result": result,
            "elapsed_ms": round((time.time() - started) * 1000, 1),
        })
        return result

    def schema_summary(self) -> str:
        """Human-readable list of registered tools, for logging/README-style
        introspection."""
        lines = []
        for name, func in self._tools.items():
            sig = inspect.signature(func)
            doc = (func.__doc__ or "").strip().splitlines()[0]
            lines.append(f"- {name}{sig}: {doc}")
        return "\n".join(lines)
