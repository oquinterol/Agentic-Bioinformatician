/**
 * Token-free check that Pi loads the GenomeAgent bridge correctly.
 *
 * Loaded next to genome-agent.ts by `genome-pi --selftest`. At session start it
 * verifies the active tool set and makes one real harness call, prints a JSON
 * report (Pi routes it to stderr in RPC mode), and shuts Pi down. No prompt is
 * ever sent to a model.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { callHarness, TOOL_NAMES } from "./genome-agent.ts";

export default function (pi: ExtensionAPI) {
	pi.on("session_start", async (_event, ctx) => {
		const active = pi.getActiveTools().sort();
		const expected = [...TOOL_NAMES].sort();
		const selftestCtx = { ...ctx, model: { provider: "selftest", id: "no-model" } } as typeof ctx;
		let harness: unknown;
		try {
			harness = await callHarness("inspect_system", {}, selftestCtx, undefined);
		} catch (err) {
			harness = { ok: false, error: String(err) };
		}
		const report = {
			active_tools: active,
			only_genome_agent_tools: JSON.stringify(active) === JSON.stringify(expected),
			builtin_tools_absent: !active.some((t) => ["bash", "edit", "write", "read"].includes(t)),
			harness_call: harness,
		};
		process.stdout.write(`GENOME_AGENT_SELFTEST ${JSON.stringify(report)}\n`);
		ctx.shutdown();
	});
}
