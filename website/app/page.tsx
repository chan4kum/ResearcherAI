import {
  ArrowRight,
  BookOpenCheck,
  Bot,
  CheckCircle2,
  CircleDot,
  Database,
  FileText,
  Gauge,
  GitBranch,
  Globe2,
  LockKeyhole,
  Network,
  Search,
  ShieldCheck,
  Sparkles,
  UploadCloud,
} from 'lucide-react';

const researchSteps = [
  'Analyze query intent',
  'Route across private docs, web, and developer references',
  'Rerank evidence',
  'Critique and repair weak claims',
];

const sourceCards = [
  {
    icon: Database,
    title: 'Private knowledge',
    text: 'Hybrid pgvector and BM25 retrieval for reports, PDFs, manuals, and internal notes.',
  },
  {
    icon: Globe2,
    title: 'Live web intelligence',
    text: 'External search augments stale knowledge when the question depends on current context.',
  },
  {
    icon: BookOpenCheck,
    title: 'Developer docs',
    text: 'Context-aware lookup for SDKs, frameworks, and implementation-specific research.',
  },
];

const capabilities = [
  'LangGraph multi-agent orchestration',
  'HyDE query expansion and hybrid retrieval',
  'Cross-encoder reranking',
  'Citation-grounded synthesis',
  'OpenAI, Gemini, and deterministic mock providers',
  'FastAPI, PostgreSQL, Kubernetes, Helm, Terraform',
];

const traceRows = [
  ['Router', 'technical + strategic', '0.94 confidence'],
  ['Retriever', '18 candidates', 'vector + BM25'],
  ['Reranker', '6 retained', 'cross-encoder'],
  ['Critic', '1 rewrite', 'claim coverage fixed'],
];

export default function Home() {
  return (
    <main className="min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-30 border-b border-border/70 bg-background/92 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
          <a href="#top" className="flex items-center gap-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-md bg-primary text-primary-foreground">
              <Search className="h-5 w-5" aria-hidden="true" />
            </span>
            <span className="text-base font-semibold">ResearcherAI</span>
          </a>
          <nav className="hidden items-center gap-7 text-sm text-muted-foreground md:flex">
            <a href="#workflow" className="hover:text-foreground">
              Workflow
            </a>
            <a href="#evidence" className="hover:text-foreground">
              Evidence
            </a>
            <a href="#platform" className="hover:text-foreground">
              Platform
            </a>
          </nav>
          <a
            href="https://github.com/chan4kum/ResearcherAI"
            className="inline-flex h-10 items-center justify-center gap-2 rounded-md border border-border bg-card px-4 text-sm font-medium text-card-foreground shadow-sm transition hover:border-primary/40 hover:text-primary"
          >
            View repo
            <ArrowRight className="h-4 w-4" aria-hidden="true" />
          </a>
        </div>
      </header>

      <section id="top" className="border-b border-border/70">
        <div className="mx-auto grid min-h-[calc(100vh-4rem)] max-w-7xl items-center gap-8 px-4 py-8 sm:px-6 lg:grid-cols-[1.02fr_0.98fr] lg:px-8">
          <div className="max-w-2xl">
            <div className="mb-5 inline-flex items-center gap-2 rounded-md border border-border bg-card px-3 py-2 text-sm text-muted-foreground">
              <CircleDot className="h-4 w-4 text-emerald-600" aria-hidden="true" />
              Autonomous deep research with auditable citations
            </div>
            <h1 className="max-w-3xl text-5xl font-semibold leading-[1.02] tracking-normal sm:text-6xl">
              ResearcherAI turns scattered knowledge into verified answers.
            </h1>
            <p className="mt-6 max-w-xl text-lg leading-8 text-muted-foreground">
              A cloud-native research workspace that routes questions across private documents, live web sources, and technical documentation, then synthesizes answers with visible evidence trails.
            </p>
            <div className="mt-8 flex flex-col gap-3 sm:flex-row">
              <a
                href="#console"
                className="inline-flex h-11 items-center justify-center gap-2 rounded-md bg-primary px-5 text-sm font-semibold text-primary-foreground shadow-sm transition hover:bg-primary/90"
              >
                Explore the console
                <ArrowRight className="h-4 w-4" aria-hidden="true" />
              </a>
              <a
                href="#platform"
                className="inline-flex h-11 items-center justify-center rounded-md border border-border bg-card px-5 text-sm font-semibold text-card-foreground transition hover:border-primary/40"
              >
                See production stack
              </a>
            </div>
          </div>

          <div id="console" className="rounded-lg border border-border bg-card p-3 shadow-2xl shadow-slate-950/10">
            <div className="rounded-md border border-border bg-background">
              <div className="flex items-center justify-between border-b border-border px-4 py-3">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <Bot className="h-4 w-4 text-primary" aria-hidden="true" />
                  Research console
                </div>
                <span className="rounded-md bg-emerald-50 px-2 py-1 text-xs font-medium text-emerald-700">
                  Ready
                </span>
              </div>
              <div className="p-4 sm:p-5">
                <label htmlFor="query" className="text-sm font-medium">
                  Research question
                </label>
                <div className="mt-2 rounded-md border border-input bg-card p-3">
                  <textarea
                    id="query"
                    defaultValue="Compare current retrieval-augmented generation strategies for enterprise technical analysis. Include trade-offs, failure modes, and implementation guidance."
                    className="min-h-28 w-full resize-none bg-transparent text-sm leading-6 outline-none"
                  />
                  <div className="mt-3 flex flex-wrap items-center justify-between gap-3 border-t border-border pt-3">
                    <div className="flex flex-wrap gap-2">
                      <span className="rounded-md bg-secondary px-2.5 py-1 text-xs font-medium text-secondary-foreground">
                        Deep research
                      </span>
                      <span className="rounded-md border border-border px-2.5 py-1 text-xs text-muted-foreground">
                        Web + docs + KB
                      </span>
                    </div>
                    <button className="inline-flex h-9 items-center gap-2 rounded-md bg-primary px-4 text-sm font-semibold text-primary-foreground">
                      Research
                      <Sparkles className="h-4 w-4" aria-hidden="true" />
                    </button>
                  </div>
                </div>

                <div className="mt-5 grid gap-3 sm:grid-cols-4">
                  {researchSteps.map((step, index) => (
                    <div key={step} className="rounded-md border border-border bg-muted/40 p-3">
                      <CheckCircle2 className="mb-3 h-4 w-4 text-emerald-600" aria-hidden="true" />
                      <p className="text-xs font-medium text-muted-foreground">Step {index + 1}</p>
                      <p className="mt-1 text-sm font-medium leading-5">{step}</p>
                    </div>
                  ))}
                </div>

                <div className="mt-5 rounded-md border border-border bg-card p-4">
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-sm font-semibold">Synthesized finding</p>
                    <span className="rounded-md border border-border px-2 py-1 text-xs text-muted-foreground">
                      6 citations
                    </span>
                  </div>
                  <p className="mt-3 text-sm leading-6 text-muted-foreground">
                    Hybrid retrieval works best when dense semantic search is paired with exact-match recall, source-aware routing, and a critic pass that rejects unsupported claims before the answer reaches the analyst.
                  </p>
                  <div className="mt-4 flex flex-wrap gap-2">
                    {['[1] pgvector chunk', '[2] BM25 excerpt', '[3] Tavily result', '[4] Context7 docs'].map((item) => (
                      <span key={item} className="rounded-md bg-accent px-2.5 py-1 text-xs font-medium text-accent-foreground">
                        {item}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="workflow" className="border-b border-border/70 px-4 py-16 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl">
          <div className="max-w-2xl">
            <p className="text-sm font-semibold uppercase text-primary">Workflow</p>
            <h2 className="mt-3 text-3xl font-semibold tracking-normal sm:text-4xl">
              Built around how analysts actually investigate.
            </h2>
          </div>
          <div className="mt-8 grid gap-4 md:grid-cols-3">
            {sourceCards.map(({ icon: Icon, title, text }) => (
              <article key={title} className="rounded-lg border border-border bg-card p-6">
                <Icon className="h-6 w-6 text-primary" aria-hidden="true" />
                <h3 className="mt-5 text-lg font-semibold">{title}</h3>
                <p className="mt-3 text-sm leading-6 text-muted-foreground">{text}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section id="evidence" className="border-b border-border/70 bg-muted/30 px-4 py-16 sm:px-6 lg:px-8">
        <div className="mx-auto grid max-w-7xl gap-8 lg:grid-cols-[0.82fr_1.18fr]">
          <div>
            <p className="text-sm font-semibold uppercase text-primary">Evidence layer</p>
            <h2 className="mt-3 text-3xl font-semibold tracking-normal sm:text-4xl">
              Provenance stays visible from search to answer.
            </h2>
            <p className="mt-4 text-base leading-7 text-muted-foreground">
              The original app already emphasizes citation badges, research history, upload flows, and telemetry. This version keeps those patterns, but gives them clearer hierarchy and less friction on smaller screens.
            </p>
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="rounded-lg border border-border bg-card p-5">
              <UploadCloud className="h-5 w-5 text-primary" aria-hidden="true" />
              <h3 className="mt-4 font-semibold">Document intake</h3>
              <p className="mt-2 text-sm leading-6 text-muted-foreground">
                Upload internal knowledge and keep it available for grounded retrieval.
              </p>
            </div>
            <div className="rounded-lg border border-border bg-card p-5">
              <GitBranch className="h-5 w-5 text-primary" aria-hidden="true" />
              <h3 className="mt-4 font-semibold">Adaptive routing</h3>
              <p className="mt-2 text-sm leading-6 text-muted-foreground">
                Send each query to the sources most likely to answer it well.
              </p>
            </div>
            <div className="rounded-lg border border-border bg-card p-5">
              <ShieldCheck className="h-5 w-5 text-primary" aria-hidden="true" />
              <h3 className="mt-4 font-semibold">Critic loop</h3>
              <p className="mt-2 text-sm leading-6 text-muted-foreground">
                Check context quality and repair thin answers before publishing.
              </p>
            </div>
            <div className="rounded-lg border border-border bg-card p-5">
              <FileText className="h-5 w-5 text-primary" aria-hidden="true" />
              <h3 className="mt-4 font-semibold">Citation trail</h3>
              <p className="mt-2 text-sm leading-6 text-muted-foreground">
                Tie claims to source chunks, URLs, and retrieval metadata.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section id="platform" className="px-4 py-16 sm:px-6 lg:px-8">
        <div className="mx-auto grid max-w-7xl gap-8 lg:grid-cols-[1fr_0.92fr]">
          <div>
            <p className="text-sm font-semibold uppercase text-primary">Production platform</p>
            <h2 className="mt-3 text-3xl font-semibold tracking-normal sm:text-4xl">
              Enterprise research infrastructure, not a prototype shell.
            </h2>
            <div className="mt-7 grid gap-3 sm:grid-cols-2">
              {capabilities.map((item) => (
                <div key={item} className="flex items-start gap-3 rounded-md border border-border bg-card p-3">
                  <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" aria-hidden="true" />
                  <span className="text-sm leading-6">{item}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="rounded-lg border border-border bg-card p-5">
            <div className="mb-5 flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold">Research trace</p>
                <p className="text-xs text-muted-foreground">Auditable pipeline snapshot</p>
              </div>
              <Gauge className="h-5 w-5 text-primary" aria-hidden="true" />
            </div>
            <div className="overflow-hidden rounded-md border border-border">
              {traceRows.map(([stage, output, method]) => (
                <div key={stage} className="grid grid-cols-3 gap-3 border-b border-border px-3 py-3 text-sm last:border-b-0">
                  <span className="font-medium">{stage}</span>
                  <span className="text-muted-foreground">{output}</span>
                  <span className="text-muted-foreground">{method}</span>
                </div>
              ))}
            </div>
            <div className="mt-5 grid gap-3 sm:grid-cols-2">
              <div className="rounded-md bg-secondary p-4">
                <LockKeyhole className="h-5 w-5 text-primary" aria-hidden="true" />
                <p className="mt-3 text-sm font-semibold">Guardrails</p>
                <p className="mt-1 text-xs leading-5 text-muted-foreground">Rate limits, cost limits, prompt-injection checks, and structured errors.</p>
              </div>
              <div className="rounded-md bg-secondary p-4">
                <Network className="h-5 w-5 text-primary" aria-hidden="true" />
                <p className="mt-3 text-sm font-semibold">Observability</p>
                <p className="mt-1 text-xs leading-5 text-muted-foreground">OpenTelemetry, Prometheus metrics, and Grafana-ready operations views.</p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
