import { ArrowRight, Bot, DatabaseZap, Menu, Route, ShieldCheck, TerminalSquare } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import baseHeroImage from "./assets/finpilot-treasury-command.png";
import revealHeroImage from "./assets/finpilot-risk-knowledge-layer.png";

const SPOTLIGHT_R = 260;

type Point = {
  x: number;
  y: number;
};

type RevealLayerProps = {
  image: string;
  cursorX: number;
  cursorY: number;
};

function RevealLayer({ image, cursorX, cursorY }: RevealLayerProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const layerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    return () => window.removeEventListener("resize", resizeCanvas);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const layer = layerRef.current;
    if (!canvas || !layer) {
      return;
    }

    let ctx: CanvasRenderingContext2D | null = null;
    try {
      ctx = canvas.getContext("2d");
    } catch {
      return;
    }

    if (!ctx || cursorX < -100 || cursorY < -100) {
      layer.style.opacity = "0";
      return;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const gradient = ctx.createRadialGradient(cursorX, cursorY, 0, cursorX, cursorY, SPOTLIGHT_R);
    gradient.addColorStop(0, "rgba(255,255,255,1)");
    gradient.addColorStop(0.4, "rgba(255,255,255,1)");
    gradient.addColorStop(0.6, "rgba(255,255,255,0.75)");
    gradient.addColorStop(0.75, "rgba(255,255,255,0.4)");
    gradient.addColorStop(0.88, "rgba(255,255,255,0.12)");
    gradient.addColorStop(1, "rgba(255,255,255,0)");

    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(cursorX, cursorY, SPOTLIGHT_R, 0, Math.PI * 2);
    ctx.fill();

    const mask = `url(${canvas.toDataURL()})`;
    layer.style.opacity = "1";
    layer.style.maskImage = mask;
    layer.style.webkitMaskImage = mask;
    layer.style.maskSize = "100% 100%";
    layer.style.webkitMaskSize = "100% 100%";
  }, [cursorX, cursorY]);

  return (
    <>
      <canvas ref={canvasRef} className="absolute inset-0 hidden pointer-events-none" aria-hidden="true" />
      <div
        ref={layerRef}
        data-testid="reveal-layer"
        aria-hidden="true"
        className="absolute inset-0 z-30 bg-cover bg-center bg-no-repeat pointer-events-none reveal-layer"
        style={{ backgroundImage: `url(${image})`, opacity: 0 }}
      />
    </>
  );
}

function App() {
  const mouse = useRef<Point>({ x: -999, y: -999 });
  const smooth = useRef<Point>({ x: -999, y: -999 });
  const rafRef = useRef<number>();
  const [cursorPos, setCursorPos] = useState<Point>({ x: -999, y: -999 });

  useEffect(() => {
    const onPointerMove = (event: PointerEvent) => {
      mouse.current = { x: event.clientX, y: event.clientY };
    };

    const animate = () => {
      smooth.current.x += (mouse.current.x - smooth.current.x) * 0.1;
      smooth.current.y += (mouse.current.y - smooth.current.y) * 0.1;
      setCursorPos({ ...smooth.current });
      rafRef.current = requestAnimationFrame(animate);
    };

    window.addEventListener("pointermove", onPointerMove);
    rafRef.current = requestAnimationFrame(animate);

    return () => {
      window.removeEventListener("pointermove", onPointerMove);
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
    };
  }, []);

  const navItems = ["Agent Runtime", "Knowledge RAG", "Safety Review", "Observability"];

  return (
    <main className="min-h-screen bg-carbon text-white font-sans">
      <nav
        aria-label="FinPilot 主导航"
        className="fixed left-0 right-0 top-0 z-[100] flex items-center justify-start px-4 py-4 sm:px-6 md:justify-between lg:px-8"
      >
        <a href="#hero" className="flex min-h-11 items-center gap-3 rounded-full focus:outline-none focus-visible:ring-2 focus-visible:ring-cyanflow">
          <span className="grid h-9 w-9 place-items-center rounded-full border border-white/20 bg-white/10 shadow-lg shadow-cyanflow/10 backdrop-blur">
            <Bot className="h-5 w-5 text-cyanflow" aria-hidden="true" />
          </span>
          <span className="text-xl font-semibold text-white">FinPilot</span>
        </a>

        <div className="hidden md:flex absolute left-1/2 -translate-x-1/2 items-center gap-1 rounded-full border border-white/15 bg-white/10 p-2 shadow-2xl shadow-black/25 backdrop-blur-xl">
          {navItems.map((item, index) => (
            <a
              key={item}
              href="#hero"
              className={`rounded-full px-4 py-2 text-sm font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-cyanflow ${
                index === 0 ? "bg-white text-ink" : "text-white/78 hover:bg-white/12 hover:text-white"
              }`}
            >
              {item}
            </a>
          ))}
        </div>

        <button
          className="hidden min-h-11 items-center gap-2 rounded-full bg-white px-5 py-2.5 text-sm font-semibold text-ink transition hover:bg-cyan-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyanflow md:inline-flex"
          type="button"
        >
          Launch CLI
          <TerminalSquare className="h-4 w-4" aria-hidden="true" />
        </button>

        <button
          aria-label="打开移动导航"
          className="mobile-nav-trigger ml-3 h-11 w-11 place-items-center rounded-full border border-cyanflow/40 bg-cyanflow text-carbon shadow-lg shadow-cyanflow/20 transition hover:bg-cyan-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-white"
          type="button"
        >
          <Menu className="h-5 w-5" aria-hidden="true" />
        </button>
      </nav>

      <section id="hero" className="relative h-screen w-full overflow-hidden bg-black" style={{ height: "100dvh" }}>
        <div
          className="absolute inset-0 z-10 bg-cover bg-center bg-no-repeat hero-zoom"
          style={{ backgroundImage: `url(${baseHeroImage})` }}
          aria-hidden="true"
        />
        <RevealLayer image={revealHeroImage} cursorX={cursorPos.x} cursorY={cursorPos.y} />

        <div className="absolute inset-0 z-40 bg-[radial-gradient(circle_at_65%_42%,rgba(53,215,208,0.08),transparent_33%),linear-gradient(90deg,rgba(3,8,13,0.82),rgba(3,8,13,0.38)_43%,rgba(3,8,13,0.7))]" />
        <div className="absolute inset-x-0 bottom-0 z-40 h-40 bg-gradient-to-t from-carbon via-carbon/55 to-transparent" />

        <div className="pointer-events-none absolute left-0 right-0 top-[14%] z-50 max-w-[335px] px-5 sm:top-[16%] sm:max-w-none lg:left-14 lg:right-auto lg:max-w-[880px] lg:px-0">
          <p className="hero-anim hero-fade mb-5 inline-flex items-center gap-2 rounded-full border border-cyanflow/25 bg-cyanflow/10 px-4 py-2 text-xs font-semibold uppercase text-cyan-100 backdrop-blur" style={{ animationDelay: "0.12s" }}>
            <Route className="h-4 w-4" aria-hidden="true" />
            Finance Agent Runtime
          </p>
          <h1 className="max-w-4xl text-4xl font-semibold leading-[1.08] text-white sm:text-6xl md:text-7xl">
            <span className="hero-anim hero-reveal block" style={{ animationDelay: "0.25s" }}>
              让财资决策穿透
            </span>
            <span className="hero-anim hero-reveal block text-cyan-100" style={{ animationDelay: "0.42s" }}>
              每一层上下文
            </span>
          </h1>
          <p className="hero-anim hero-fade mt-6 max-w-2xl text-base leading-8 text-white/78 sm:text-lg" style={{ animationDelay: "0.62s" }}>
            FinPilot 将共享财务知识、用户记忆、风险审查与审计链路编排进同一个可追溯的 Finance Agent Runtime。
          </p>

          <div className="pointer-events-auto hero-anim hero-fade mt-8 flex flex-col gap-3 sm:flex-row" style={{ animationDelay: "0.78s" }}>
            <button
              className="inline-flex min-h-12 w-full items-center justify-center gap-2 rounded-full bg-cyanflow px-7 py-3 text-sm font-semibold text-carbon shadow-xl shadow-cyanflow/20 transition hover:scale-[1.02] hover:bg-cyan-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-white active:scale-95 sm:w-auto"
              type="button"
            >
              查看运行链路
              <ArrowRight className="h-4 w-4" aria-hidden="true" />
            </button>
            <button
              className="inline-flex min-h-12 w-full items-center justify-center gap-2 rounded-full border border-white/18 bg-white/10 px-7 py-3 text-sm font-semibold text-white backdrop-blur transition hover:scale-[1.02] hover:bg-white/18 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyanflow active:scale-95 sm:w-auto"
              type="button"
            >
              启动 CLI
              <TerminalSquare className="h-4 w-4" aria-hidden="true" />
            </button>
          </div>
        </div>

        <aside className="hero-anim hero-fade pointer-events-none absolute bottom-12 left-5 z-50 hidden max-w-[300px] rounded-2xl border border-white/12 bg-black/22 p-4 text-sm leading-7 text-white/72 backdrop-blur-md sm:block lg:left-14" style={{ animationDelay: "0.9s" }}>
          <div className="mb-3 flex items-center gap-2 text-cyan-100">
            <DatabaseZap className="h-4 w-4" aria-hidden="true" />
            <span className="text-xs font-semibold uppercase">Shared Knowledge</span>
          </div>
          共享知识库统一进入 BM25、向量检索与重排链路，回答保留来源、分数与审计上下文。
        </aside>

        <aside className="hero-anim hero-fade pointer-events-none absolute bottom-8 left-5 z-50 w-[calc(100vw-2.5rem)] max-w-[335px] rounded-2xl border border-white/12 bg-black/25 p-4 text-sm leading-7 text-white/76 backdrop-blur-md sm:bottom-12 sm:left-auto sm:right-8 sm:max-w-[330px] lg:right-14" style={{ animationDelay: "1s" }}>
          <div className="mb-3 flex items-center gap-2 text-risk">
            <ShieldCheck className="h-4 w-4" aria-hidden="true" />
            <span className="text-xs font-semibold uppercase">Safety & Audit</span>
          </div>
          移动光标探索隐藏层：输入审查、工具风险、结果复核与最终回答审查会在同一条审计链中显影。
        </aside>

        <div className="pointer-events-none absolute bottom-6 left-1/2 z-50 hidden -translate-x-1/2 rounded-full border border-white/12 bg-white/8 px-4 py-2 text-xs text-white/60 backdrop-blur md:block">
          移动光标，揭示 FinPilot 的风险与知识编排层
        </div>
      </section>
    </main>
  );
}

export default App;
