import os
import xml.etree.ElementTree as ET

def create_ultra_clean_rag_architecture_diagram():
    # Define root mxfile
    mxfile = ET.Element("mxfile", {
        "host": "Electron",
        "modified": "2026-06-19T16:00:00.000Z",
        "agent": "5.0",
        "version": "20.0.0",
        "type": "device"
    })
    
    diagram = ET.SubElement(mxfile, "diagram", {
        "id": "diag_rag_ultra_clean",
        "name": "RAG Clean Architecture"
    })
    
    # Grid and page dimensions
    mxGraphModel = ET.SubElement(diagram, "mxGraphModel", {
        "dx": "1200",
        "dy": "1000",
        "grid": "1",
        "gridSize": "10",
        "guides": "1",
        "tooltips": "1",
        "connect": "1",
        "arrows": "1",
        "fold": "1",
        "page": "1",
        "pageScale": "1",
        "pageWidth": "1600",
        "pageHeight": "1150",
        "background": "#0F172A", # Deep slate background
        "math": "0",
        "shadow": "0"
    })
    
    root = ET.SubElement(mxGraphModel, "root")
    
    # Base cells
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})
    
    # Styling helper for custom containers
    def add_layer_container(id_str, value, x, y, w, h, stroke_color, text_color, fill_color="#090D16"):
        style = (
            f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill_color};strokeColor={stroke_color};"
            f"fontColor={text_color};fontSize=12;fontStyle=1;align=left;verticalAlign=top;"
            f"spacingTop=12;spacingLeft=18;strokeWidth=2;arcSize=4;connectable=0;"
        )
        cell = ET.SubElement(root, "mxCell", {
            "id": id_str,
            "value": value,
            "style": style,
            "vertex": "1",
            "parent": "1"
        })
        ET.SubElement(cell, "mxGeometry", {
            "x": str(x),
            "y": str(y),
            "width": str(w),
            "height": str(h),
            "as": "geometry"
        })
        return cell

    # Helper for normal node blocks
    def add_node(id_str, value, x, y, w, h, style):
        cell = ET.SubElement(root, "mxCell", {
            "id": id_str,
            "value": value,
            "style": style,
            "vertex": "1",
            "parent": "1"
        })
        ET.SubElement(cell, "mxGeometry", {
            "x": str(x),
            "y": str(y),
            "width": str(w),
            "height": str(h),
            "as": "geometry"
        })
        return cell

    # Helper for connectors with custom routing ports & clean light-themed labels
    def add_edge(id_str, source_id, target_id, color, label="", exit_port="", entry_port=""):
        style = (
            f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
            f"strokeColor={color};strokeWidth=1.5;fillColor=none;"
            f"endArrow=classic;endFill=1;fontSize=9;fontColor=#0F172A;"
            f"labelBackgroundColor=#F8FAFC;labelBorderColor=#64748B;fontStyle=1;" # Light badge theme
        )
        if exit_port:
            style += f"{exit_port};"
        if entry_port:
            style += f"{entry_port};"
            
        cell = ET.SubElement(root, "mxCell", {
            "id": id_str,
            "value": label,
            "style": style,
            "edge": "1",
            "parent": "1",
            "source": source_id,
            "target": target_id
        })
        ET.SubElement(cell, "mxGeometry", {
            "relative": "1",
            "as": "geometry"
        })
        return cell

    # ==================== 1. CONTAINERS & BACKDROPS ====================
    # Title Block
    add_node(
        "title_block", 
        "MULTIMODAL DOCUMENT INTELLIGENCE RAG ARCHITECTURE\n"
        "Strict Layered Enterprise Middleware Design with Independent QA & Diagnostics Subsystems",
        360, 15, 880, 60,
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#1E293B;strokeColor=#475569;fontColor=#F8FAFC;align=center;strokeWidth=2;fontSize=14;fontStyle=1;arcSize=8;"
    )

    # QA & Validation Container (Left Sidebar)
    add_layer_container(
        "validation_subsystem",
        "QA & VALIDATION CO-PROCESS SYSTEM",
        50, 100, 280, 750,
        stroke_color="#D97706", text_color="#FBBF24", fill_color="#0F1319"
    )

    # Observability Subsystem Container (Right Sidebar)
    add_layer_container(
        "observability_subsystem",
        "DIAGNOSTICS & SYSTEM TELEMETRY",
        1270, 100, 280, 750,
        stroke_color="#E11D48", text_color="#FB7185", fill_color="#0F1319"
    )

    # Horizontal Layers (Middle)
    add_layer_container(
        "l1_presentation",
        "L1: PRESENTATION & CLIENT LAYER",
        360, 100, 880, 110,
        stroke_color="#0284C7", text_color="#38BDF8"
    )

    add_layer_container(
        "l2_api",
        "L2: REST API GATEWAY & ROUTING SERVICE",
        360, 240, 880, 110,
        stroke_color="#4F46E5", text_color="#818CF8"
    )

    add_layer_container(
        "l3_cognitive",
        "L3: COGNITIVE MIDDLEWARE & ORCHESTRATION PIPELINES",
        360, 380, 880, 140,
        stroke_color="#7C3AED", text_color="#A78BFA"
    )

    add_layer_container(
        "l4_ai",
        "L4: MULTIMODAL AI INFERENCE ENGINE & DRIVERS",
        360, 550, 880, 140,
        stroke_color="#DB2777", text_color="#F472B6"
    )

    add_layer_container(
        "l5_storage",
        "L5: SYSTEM PERSISTENCE & DATA STORAGE FABRIC",
        360, 720, 880, 130,
        stroke_color="#059669", text_color="#34D399"
    )

    # Roadmap Container (Bottom)
    add_layer_container(
        "roadmap",
        "FUTURE ARCHITECTURAL ROADMAP & EVOLUTION PATHWAY",
        50, 880, 1500, 190,
        stroke_color="#475569", text_color="#94A3B8", fill_color="#0F1319"
    )


    # ==================== 2. QA & VALIDATION SYSTEM NODES ====================
    style_val_node = (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#18120A;strokeColor=#D97706;fontColor=#FDE68A;"
        "fontSize=10;align=left;spacingLeft=12;arcSize=6;strokeWidth=1.2;"
    )

    add_node(
        "val_pytest",
        "<b>Automated PyTest Suite</b><br/>"
        "• <i>test_chunking.py</i> (Partitions)<br/>"
        "• <i>test_embeddings.py</i> (Vectors)<br/>"
        "• <i>test_ocr.py</i> (Paddle/EasyOCR fallback)<br/>"
        "• <i>test_whisper.py / test_piper.py</i>",
        65, 155, 250, 95, style_val_node
    )

    add_node(
        "val_router_apis",
        "<b>Validation API Router (validation.py)</b><br/>"
        "• <i>/image-summary</i>: Gemini verify<br/>"
        "• <i>/ocr</i>: Baseline extraction checks<br/>"
        "• <i>/visual-understanding</i>: Fusion<br/>"
        "• <i>/table-extraction</i>: pdfplumber<br/>"
        "• <i>/pdf-images</i>: Extract binaries",
        65, 270, 250, 120, style_val_node
    )

    add_node(
        "val_tabular_full",
        "<b>Tabular & Full PDF Analysis</b><br/>"
        "• <i>/tabular-data-analysis</i>:<br/>"
        "  Parses PDFs, XLSX, XLS, CSV to JSON<br/>"
        "• <i>/pdf-full-analysis</i>:<br/>"
        "  Parallel pages, tables, images auditor",
        65, 410, 250, 95, style_val_node
    )

    add_node(
        "val_temp_store",
        "<b>Validation Temp Cache</b><br/>"
        "• Path: <i>data/temp/validation_images/</i><br/>"
        "• Stores transient image extractions<br/>"
        "• Exposes `/image/{name}` preview path",
        65, 525, 250, 80, style_val_node
    )

    add_node(
        "val_eval",
        "<b>QA Grounding Evaluator</b><br/>"
        "• Response-Context token alignment<br/>"
        "• Hallucination detection validator<br/>"
        "• RAG answer exactness verification",
        65, 625, 250, 85, style_val_node
    )

    add_node(
        "val_manual",
        "<b>Manual Testing Harness</b><br/>"
        "• <i>test_manual_validation.py</i><br/>"
        "• Simulates multi-page client uploads<br/>"
        "• Validates raw stream payloads",
        65, 730, 250, 85, style_val_node
    )


    # ==================== 3. OBSERVABILITY SYSTEM NODES ====================
    style_obs_node = (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#1B080C;strokeColor=#E11D48;fontColor=#FFE4E6;"
        "fontSize=10;align=left;spacingLeft=12;arcSize=6;strokeWidth=1.2;"
    )

    add_node(
        "obs_diagnostics",
        "<b>Startup Check (log_system_check)</b><br/>"
        "• Python version check & runtime diagnostics<br/>"
        "• CUDA / GPU capability active detection<br/>"
        "• Connection: Gemini, SQLite, Qdrant DB<br/>"
        "• Audio drivers active verification",
        1285, 155, 250, 95, style_obs_node
    )

    add_node(
        "obs_performance",
        "<b>Performance Latency Profiler</b><br/>"
        "• Records end-to-end processing speeds<br/>"
        "• Logs text extraction & OCR delay times<br/>"
        "• Tracks TTS / STT model conversion time<br/>"
        "• Telemetry logging daemon",
        1285, 270, 250, 95, style_obs_node
    )

    add_node(
        "obs_logger_daemon",
        "<b>Structured Logging Daemon (logger.py)</b><br/>"
        "• <b>app.log</b>: Core application telemetry<br/>"
        "• <b>ingestion.log</b>: Ingestion page statistics<br/>"
        "• <b>retrieval.log</b>: Vector search & score logs<br/>"
        "• <b>chat.log</b>: Prompt context & LLM outputs<br/>"
        "• <b>errors.log</b>: Structured exception tracebacks",
        1285, 385, 250, 200, style_obs_node
    )

    add_node(
        "obs_alerts",
        "<b>System Alerts & Fault Auditing</b><br/>"
        "• Throttling & API retry controller<br/>"
        "• SQLite transaction lock monitor<br/>"
        "• Auto-fallback on Gemini timeout",
        1285, 605, 250, 85, style_obs_node
    )

    add_node(
        "obs_resources",
        "<b>Hardware Resource Monitor</b><br/>"
        "• VRAM usage for Whisper/Piper inference<br/>"
        "• RAM thresholds tracking<br/>"
        "• Host disk cache utilization metrics",
        1285, 710, 250, 85, style_obs_node
    )


    # ==================== 4. LAYER NODES (MIDDLE) ====================
    # L1: Presentation
    style_l1_node = (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#0A1520;strokeColor=#0284C7;fontColor=#E0F2FE;"
        "fontSize=10;align=center;arcSize=8;strokeWidth=1.5;"
    )
    add_node(
        "l1_cli_client",
        "<b>Interactive CLI Client Shell (cli.py)</b><br/>Voice & Text interface, sounddevice audio recorder & wav player",
        400, 140, 360, 50, style_l1_node
    )
    add_node(
        "l1_external_client",
        "<b>External API / Downstream Client</b><br/>Downstream business applications, web interfaces & API consumers",
        820, 140, 360, 50, style_l1_node
    )

    # L2: API Gateway
    style_l2_node = (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#0B0B20;strokeColor=#4F46E5;fontColor=#EEF2FF;"
        "fontSize=10;align=center;arcSize=8;strokeWidth=1.5;"
    )
    add_node(
        "l2_fastapi",
        "<b>FastAPI Application Instance (app.py)</b><br/>CORS gateway, router mappings, payload limits validation (<50MB)",
        400, 280, 360, 50, style_l2_node
    )
    add_node(
        "l2_uvicorn",
        "<b>Uvicorn ASGI Server Instance (server.py)</b><br/>Web worker processes, event loops, process binding configuration",
        820, 280, 360, 50, style_l2_node
    )

    # L3: Cognitive Middleware
    style_l3_core = (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#130A2B;strokeColor=#7C3AED;fontColor=#F5F3FF;"
        "fontSize=10;align=center;arcSize=8;strokeWidth=2;fontStyle=1;"
    )
    style_l3_sub = (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#070716;strokeColor=#7C3AED;fontColor=#F5F3FF;"
        "fontSize=10;align=center;arcSize=8;strokeWidth=1.5;"
    )
    add_node(
        "l3_ingestion_engine",
        "<b>Ingestion Pipeline Engine<br/>(ingestion.py)</b><br/>Splits PDF, triggers PaddleOCR / pdfplumber parser, parses table blocks & generates embeddings",
        390, 430, 240, 70, style_l3_core
    )
    add_node(
        "l3_query_engine",
        "<b>RAG Query Engine<br/>(query_engine.py)</b><br/>Triggers vector search, ranks retrieved contexts, builds LLM prompts & returns grounded response",
        680, 430, 240, 70, style_l3_core
    )
    add_node(
        "l3_audio_orchestrator",
        "<b>Audio Subprocess Drivers<br/>(speech_to_text.py, text_to_speech.py)</b><br/>Whisper STT file transcriber wrapper & Piper TTS ONNX audio synthesis coordinator",
        970, 430, 240, 70, style_l3_sub
    )

    # L4: AI Inference & Drivers
    style_l4_core = (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#28081A;strokeColor=#DB2777;fontColor=#FDF2F8;"
        "fontSize=10;align=center;arcSize=8;strokeWidth=2;fontStyle=1;"
    )
    style_l4_sub = (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#0D040A;strokeColor=#DB2777;fontColor=#FDF2F8;"
        "fontSize=10;align=center;arcSize=8;strokeWidth=1.5;"
    )
    add_node(
        "l4_gemini_driver",
        "<b>Google Gemini API Driver</b><br/>"
        "• gemini-3.1-flash-lite<br/>"
        "• gemini-embedding-2<br/>"
        "• Gemini Multimodal Vision",
        380, 590, 185, 80, style_l4_core
    )
    add_node(
        "l4_ocr_engines",
        "<b>Local OCR Subsystems</b><br/>"
        "• PaddleOCR (primary)<br/>"
        "• EasyOCR (fallback reader)<br/>"
        "• Layout coords merger",
        590, 590, 185, 80, style_l4_sub
    )
    add_node(
        "l4_pdf_parsers",
        "<b>Native PDF Extractors</b><br/>"
        "• pdfplumber (table layout)<br/>"
        "• pypdf (binary images)<br/>"
        "• Text stream parsers",
        800, 590, 185, 80, style_l4_sub
    )
    add_node(
        "l4_local_inference",
        "<b>Local Audio Inference</b><br/>"
        "• Whisper STT Local model<br/>"
        "• Piper ONNX TTS Engine<br/>"
        "• lessac-medium voice",
        1010, 590, 185, 80, style_l4_sub
    )

    # L5: Storage & Persistence
    style_l5_db = (
        "shape=cylinder;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;size=15;"
        "fillColor=#011E11;strokeColor=#059669;fontColor=#ECFDF5;fontSize=10;align=center;fontStyle=1;strokeWidth=2;"
    )
    style_l5_cache = (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#030A07;strokeColor=#059669;fontColor=#ECFDF5;"
        "fontSize=10;align=center;arcSize=8;strokeWidth=1.5;"
    )
    add_node(
        "l5_sqlite_db",
        "<b>SQLite relational DB<br/>(rag_tool.db)</b><br/>"
        "Document index, page segments metadata, chunk text & table schemas",
        400, 750, 220, 75, style_l5_db
    )
    add_node(
        "l5_qdrant_db",
        "<b>Qdrant Vector DB<br/>(data/qdrant/)</b><br/>"
        "Local vector collections, search index & semantic embeddings store",
        690, 750, 220, 75, style_l5_db
    )
    add_node(
        "l5_filesystem_cache",
        "<b>Local Cache Directory<br/>(data/temp/)</b><br/>"
        "Uploaded raw documents, page fragments & cached audio outputs",
        980, 750, 220, 75, style_l5_cache
    )


    # ==================== 5. ROADMAP TIMELINE NODES (BOTTOM) ====================
    style_road_node = (
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#1E293B;strokeColor=#475569;fontColor=#F1F5F9;"
        "fontSize=9.5;align=left;spacingLeft=12;arcSize=6;strokeWidth=1.5;"
    )
    
    add_node(
        "road_phase_1",
        "<b>PHASE 1: DUAL-DUPLEX VOICE STREAMING</b><br/>"
        "• Establish WebSockets communication channels.<br/>"
        "• Real-time duplex audio loop (STT/TTS streaming).<br/>"
        "• User interrupt detection (speech barge-in).",
        88, 935, 320, 110, style_road_node
    )
    
    add_node(
        "road_phase_2",
        "<b>PHASE 2: HYBRID DENSE-SPARSE SEMANTIC SEARCH</b><br/>"
        "• Integrate BM25 sparse keyword indices inside SQLite.<br/>"
        "• Implement Reciprocal Rank Fusion (RRF) rank merger.<br/>"
        "• Combine literal keyword hits with semantic dense vectors.",
        456, 935, 320, 110, style_road_node
    )
    
    add_node(
        "road_phase_3",
        "<b>PHASE 3: GRAPH RAG CONTEXT REINFORCEMENT</b><br/>"
        "• Run entity-relation extraction on ingested chunks.<br/>"
        "• Construct entity knowledge graph inside SQLite DB.<br/>"
        "• Query local subgraphs to enrich context templates.",
        824, 935, 320, 110, style_road_node
    )
    
    add_node(
        "road_phase_4",
        "<b>PHASE 4: AGENTIC SELF-CORRECTION QA FEEDBACK</b><br/>"
        "• Deploy lightweight auditing agents on response loops.<br/>"
        "• Verify facts against original source document chunks.<br/>"
        "• Automatically rebuild answers failing validation threshold.",
        1192, 935, 320, 110, style_road_node
    )


    # ==================== 6. CLEAN RE-STRUCTURED STRATIFIED ARROWS ====================
    # Style Color codes
    c_indigo = "#818CF8"  # Command Flow (Requests)
    c_sky = "#38BDF8"     # Audio I/O Flow
    c_pink = "#F472B6"    # AI Inference Drivers
    c_emerald = "#34D399" # Data Persistence
    c_amber = "#FBBF24"   # QA / Test validation link
    c_rose = "#FB7185"    # System telemetry / logs path

    # Layer 1 -> Layer 2 (Presentation -> API Gateway)
    add_edge("edge_cli_to_app", "l1_cli_client", "l2_fastapi", c_indigo, "REST API")
    add_edge("edge_web_to_app", "l1_external_client", "l2_fastapi", c_indigo, "HTTP Request")
    
    # Layer 2 -> Layer 3 (API Gateway -> Cognitive Middleware)
    add_edge("edge_app_to_ingest", "l2_fastapi", "l3_ingestion_engine", c_indigo, "Ingest PDF")
    add_edge("edge_app_to_query", "l2_fastapi", "l3_query_engine", c_indigo, "Query Text")
    # Voice CLI connects straight to Audio Subprocess wrapper, routed around Uvicorn boxes via right margin
    add_edge("edge_cli_to_audio", "l1_cli_client", "l3_audio_orchestrator", c_sky, "Audio Stream", "exitX=1;exitY=0.5", "entryX=1;entryY=0.5")

    # Layer 3 -> Layer 4 (Cognitive Middleware -> AI Inference Drivers)
    # Strictly layered: connects to the component directly below it
    add_edge("edge_ingest_to_parsers", "l3_ingestion_engine", "l4_pdf_parsers", c_pink, "Parse Layout")
    add_edge("edge_ingest_to_ocr", "l3_ingestion_engine", "l4_ocr_engines", c_pink, "OCR Scan")
    add_edge("edge_query_to_gemini", "l3_query_engine", "l4_gemini_driver", c_pink, "LLM Request")
    add_edge("edge_audio_to_local_inf", "l3_audio_orchestrator", "l4_local_inference", c_pink, "STT / TTS")

    # Layer 4 -> Layer 5 (AI Inference Drivers -> Persistence Layer)
    # The models and parsers themselves save their respective extractions into databases.
    # No lines skip from L3 to L5 anymore, keeping L4 completely uncluttered.
    add_edge("edge_parsers_to_sqlite", "l4_pdf_parsers", "l5_sqlite_db", c_emerald, "Save Text")
    add_edge("edge_ocr_to_sqlite", "l4_ocr_engines", "l5_sqlite_db", c_emerald, "Save Text")
    add_edge("edge_gemini_to_qdrant", "l4_gemini_driver", "l5_qdrant_db", c_emerald, "Index Vectors")
    add_edge("edge_audio_to_cache", "l4_local_inference", "l5_filesystem_cache", c_emerald, "Cache WAV")

    # Subsystems connections (amber & rose) - reduced to a single entry/exit flow to keep diagram clean.
    add_edge("edge_val_router_to_api", "val_router_apis", "l2_fastapi", c_amber, "Verify API")
    add_edge("edge_app_to_logger", "l2_fastapi", "obs_logger_daemon", c_rose, "Traffic Logs")

    # Generate XML tree
    tree = ET.ElementTree(mxfile)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RAG_Architecture.drawio")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Draw.io Stratified clean XML successfully created and saved to: {output_path}")

if __name__ == "__main__":
    create_ultra_clean_rag_architecture_diagram()
