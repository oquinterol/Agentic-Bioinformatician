/**
 * GenomeAgent bridge for Pi (verified against Pi 0.87.1).
 *
 * Deliberately logic-free: every tool forwards its arguments as JSON to
 *   genome-agent tool <operation> --project <dir> --actor llm:<provider>/<model>
 * and returns the JSON answer. Validation, resource policy, execution, state
 * and provenance all live in the Python harness, where they are tested.
 *
 * Environment:
 *   GENOME_AGENT_BIN      genome-agent executable (default: "genome-agent" on PATH)
 *   GENOME_AGENT_PROJECT  project directory (default: Pi's working directory)
 *
 * Run it through ./genome-pi, which also disables Pi's built-in tools.
 */

import { spawn } from "node:child_process";
import { StringEnum } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type, type TSchema } from "typebox";

const MAX_MODEL_TEXT = 40_000; // characters of JSON returned to the model

const FreeObject = (description: string) =>
	Type.Object({}, { additionalProperties: true, description });

const TOOLS: {
	name: string;
	label: string;
	description: string;
	snippet: string;
	guidelines?: string[];
	parameters: TSchema;
}[] = [
	{
		name: "inspect_system",
		label: "Inspect system",
		description:
			"Authoritative inventory of this machine (CPU, RAM, disk, detected tools) and the resource budget every job must fit in. Never assume resources; read them here.",
		snippet: "Machine inventory and the resource budget jobs must fit in",
		parameters: Type.Object({}),
	},
	{
		name: "list_tools",
		label: "List tools",
		description:
			"Bioinformatics tools known to GenomeAgent, in preference order, with purpose, input/output types, parameter JSON schema, and whether each is available on this machine.",
		snippet: "Available bioinformatics tools and their parameter schemas",
		parameters: Type.Object({}),
	},
	{
		name: "project_status",
		label: "Project status",
		description:
			"Objective, biological context, datasets, budget, jobs (with failures and rejection reasons), results, recent decisions and metrics of the current project.",
		snippet: "Current scientific state of the project",
		guidelines: [
			"Call project_status before deciding the next step, and after every run_tool, instead of relying on memory.",
		],
		parameters: Type.Object({}),
	},
	{
		name: "assess_tool",
		label: "Assess tool",
		description:
			"Estimate peak CPU/RAM/disk for a tool on the project data and check it against the budget, trying fewer threads if needed. Read-only: nothing is executed.",
		snippet: "Check whether a tool fits the resource budget, without running it",
		guidelines: ["Use assess_tool on every candidate before calling run_tool."],
		parameters: Type.Object({
			tool: Type.String({ description: "Tool name from list_tools" }),
			params: Type.Optional(FreeObject("Tool parameters matching its params_schema")),
			inputs: Type.Optional(
				Type.Array(Type.String(), { description: "Input files; default: all project datasets" }),
			),
			genome_size_bp: Type.Optional(
				Type.Integer({ minimum: 1, description: "Default: project biological context" }),
			),
		}),
	},
	{
		name: "run_tool",
		label: "Run tool",
		description:
			"Execute a tool through the GenomeAgent harness. The request is validated (tool, params, inputs, CPU/RAM vs budget and vs estimate) and may be rejected with reasons. Outputs go to the project's runs/ directory. Blocks until the job finishes.",
		snippet: "Execute a validated bioinformatics job and record the decision",
		guidelines: [
			"run_tool requires a scientific reason and the alternatives you considered; they become the auditable decision record.",
			"Use the cpus and ram_gb from assess_tool's estimate when calling run_tool.",
			"If run_tool reports a failed or rejected job, do not repeat the identical request; replan.",
		],
		parameters: Type.Object({
			tool: Type.String(),
			params: Type.Optional(FreeObject("Tool parameters matching its params_schema")),
			inputs: Type.Optional(Type.Array(Type.String())),
			genome_size_bp: Type.Optional(Type.Integer({ minimum: 1 })),
			cpus: Type.Integer({ minimum: 1 }),
			ram_gb: Type.Number({ exclusiveMinimum: 0 }),
			timeout_s: Type.Optional(Type.Number({ exclusiveMinimum: 0 })),
			reason: Type.String({ minLength: 1, description: "Scientific justification" }),
			alternatives_considered: Type.Optional(Type.Array(Type.String())),
			evidence: Type.Optional(FreeObject("Facts supporting the decision")),
		}),
	},
	{
		name: "record_decision",
		label: "Record decision",
		description:
			"Record a scientific decision that does not launch a job: stopping, concluding the objective is unreachable, choosing a strategy, interpreting a result.",
		snippet: "Record an auditable decision (e.g. stop and why)",
		guidelines: [
			"When the objective is reached or cannot be reached responsibly, call record_decision with decision 'stop' and the reason before ending.",
		],
		parameters: Type.Object({
			decision: Type.String({ minLength: 1 }),
			reason: Type.String({ minLength: 1 }),
			evidence: Type.Optional(FreeObject("Facts supporting the decision")),
			alternatives_considered: Type.Optional(Type.Array(Type.String())),
		}),
	},
	{
		name: "add_dataset",
		label: "Add dataset",
		description:
			"Register an existing sequencing file (anywhere on this machine, read-only) as a project dataset.",
		snippet: "Register an existing reads file as a project dataset",
		parameters: Type.Object({
			path: Type.String({ description: "Path to an existing, readable file" }),
			kind: StringEnum(["pacbio_hifi", "ont", "illumina", "hic", "rnaseq", "other"] as const),
		}),
	},
];

export const TOOL_NAMES = new Set(TOOLS.map((t) => t.name));

function actorFor(ctx: ExtensionContext): string {
	const m = ctx.model;
	return m ? `llm:${m.provider}/${m.id}` : "llm:unknown";
}

export function callHarness(
	operation: string,
	args: unknown,
	ctx: ExtensionContext,
	signal: AbortSignal | undefined,
): Promise<{ ok: boolean; result?: unknown; error?: string }> {
	const bin = process.env.GENOME_AGENT_BIN ?? "genome-agent";
	const project = process.env.GENOME_AGENT_PROJECT ?? ctx.cwd;
	const argv = ["tool", operation, "--project", project, "--actor", actorFor(ctx)];

	return new Promise((resolve, reject) => {
		const child = spawn(bin, argv, { stdio: ["pipe", "pipe", "pipe"], signal });
		let stdout = "";
		let stderr = "";
		child.stdout.on("data", (d) => (stdout += d));
		child.stderr.on("data", (d) => (stderr += d));
		child.on("error", (err) => reject(new Error(`could not run ${bin}: ${err.message}`)));
		child.on("close", (code) => {
			try {
				resolve(JSON.parse(stdout));
			} catch {
				reject(
					new Error(`genome-agent exited with ${code} without a JSON answer: ${stderr.slice(-2000)}`),
				);
			}
		});
		child.stdin.end(JSON.stringify(args ?? {}));
	});
}

export default function (pi: ExtensionAPI) {
	for (const t of TOOLS) {
		pi.registerTool({
			name: t.name,
			label: t.label,
			description: t.description,
			promptSnippet: t.snippet,
			promptGuidelines: t.guidelines,
			parameters: t.parameters,
			// Jobs mutate the shared project state; never run them concurrently.
			executionMode: "sequential",
			async execute(_toolCallId, params, signal, _onUpdate, ctx) {
				const response = await callHarness(t.name, params, ctx, signal);
				if (!response.ok) throw new Error(response.error ?? "unknown GenomeAgent error");
				let text = JSON.stringify(response.result, null, 1);
				if (text.length > MAX_MODEL_TEXT) {
					text = `${text.slice(0, MAX_MODEL_TEXT)}\n... [truncated; call project_status for a summary]`;
				}
				return { content: [{ type: "text", text }], details: response.result };
			},
		});
	}

	// Defence in depth: even if another extension or a setting re-enables other
	// tools, the model may only call GenomeAgent tools in this session.
	pi.on("tool_call", async (event) => {
		if (!TOOL_NAMES.has(event.toolName)) {
			return {
				block: true,
				reason: `${event.toolName} is not a GenomeAgent tool; only harness tools are allowed`,
			};
		}
		return undefined;
	});
}
