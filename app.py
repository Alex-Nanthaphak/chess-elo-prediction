import io, os, math, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, chess, chess.pgn, chess.engine, shap
import streamlit as st

st.set_page_config(page_title="Prédiction du classement Elo aux échecs", page_icon="♟️", layout="wide")

MODEL_PATH = (
    "models/elo_predictor_pipeline.joblib"
    if os.path.exists("models/elo_predictor_pipeline.joblib")
    else "elo_predictor_pipeline.joblib"
)
DEFAULT_STOCKFISH = (
    os.getenv("STOCKFISH_PATH")
    or shutil.which("stockfish")
    or ("/usr/games/stockfish" if os.path.exists("/usr/games/stockfish") else "")
    or r"C:\Users\Alex\Desktop\Projet\stockfish\stockfish-windows-x86-64-universal.exe"
)

MIN_PLIES = 20
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
RES_MAP = {"1-0": (1.0, 0.0), "0-1": (0.0, 1.0), "1/2-1/2": (0.5, 0.5)}
RESULT_LABELS = {1.0: "Victoire", 0.5: "Nulle", 0.0: "Défaite"}
CADENCE_ORDER = ["Bullet", "Blitz", "Rapide", "Classique"]
KEY_STD_COLS = ["acpl", "acpl_opening", "accuracy_pct", "blunder_rate", "total_plies", "avg_time_per_move"]
TERM_CATEGORIES = ["Nulle", "Abandon", "Temps écoulé", "Échec et mat"]
mean_nan = lambda x: float(np.mean(x)) if x else np.nan


# --- Fonctions métier (à importer de src/features.py après modularisation) ---
def parse_time_control(tc_str: str):
    if not tc_str or tc_str == "-" or "/" in tc_str:
        return np.nan, np.nan, np.nan, "Unknown"
    try:
        parts = tc_str.split("+")
        base_s, inc_s = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        return np.nan, np.nan, np.nan, "Unknown"
    dur = base_s + 40 * inc_s
    cat = "Bullet" if dur < 180 else "Blitz" if dur < 480 else "Rapide" if dur < 1500 else "Classique"
    return base_s, inc_s, dur, cat


def cp_to_win_pct(cp: float) -> float:
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


def win_loss_to_accuracy(win_loss: float) -> float:
    return float(np.clip(103.1668 * math.exp(-0.04354 * win_loss) - 3.1669, 0.0, 100.0))


def count_material(board: chess.Board, color: bool) -> int:
    return sum(len(board.pieces(pt, color)) * v for pt, v in PIECE_VALUES.items())


def extract_features_from_game(game: chess.pgn.Game, game_id: int = 0):
    h = game.headers
    if h.get("Variant", "Standard") != "Standard" or h.get("Result") not in RES_MAP:
        return None
    base_t, inc, _, cat = parse_time_control(h.get("TimeControl", ""))
    if cat == "Unknown":
        return None

    board = game.board()
    stats = {
        c: dict(cpls=[], open=[], mid=[], end=[], win=[], accs=[], evals=[], times=[], open_t=[],
                blunders=0, mistakes=0, inacc=0, trouble=0, did_castle=0, castle_ply=np.nan, last_clk=float(base_t))
        for c in (chess.WHITE, chess.BLACK)
    }
    prev_eval, ply, eval_count = 20.0, 0, 0

    for node in game.mainline():
        move, turn = node.move, board.turn
        s = stats[turn]
        ply += 1
        if board.is_castling(move):
            s["did_castle"], s["castle_ply"] = 1, ply
        board.push(move)

        if (ev := node.eval()) is not None:
            eval_count += 1
            curr_eval = float(np.clip(ev.white().score(mate_score=1000), -1000, 1000))
            s["evals"].append(curr_eval)
            sign = 1.0 if turn == chess.WHITE else -1.0
            cpl = max(0.0, sign * (prev_eval - curr_eval))
            win_loss = max(0.0, cp_to_win_pct(sign * prev_eval) - cp_to_win_pct(sign * curr_eval))
            s["cpls"].append(cpl)
            s["win"].append(win_loss)
            s["accs"].append(win_loss_to_accuracy(win_loss))
            s["open" if ply <= 20 else "mid" if ply <= 60 else "end"].append(cpl)
            if cpl >= 200: s["blunders"] += 1
            elif cpl >= 100: s["mistakes"] += 1
            elif cpl >= 50: s["inacc"] += 1
            prev_eval = curr_eval

        if (clk := node.clock()) is not None:
            if ply > 2:
                spent = max(0.0, s["last_clk"] - clk + inc)
                s["times"].append(spent)
                if ply <= 20:
                    s["open_t"].append(spent)
            s["trouble"] += clk < min(10.0, 0.1 * base_t)
            s["last_clk"] = clk

    if ply < MIN_PLIES or eval_count < ply - 1:
        return None

    raw_term = h.get("Termination", "").lower()
    is_time = any(k in raw_term for k in ("time forfeit", "on time", "au temps", "temps écoulé"))
    term = "Temps écoulé" if is_time else "Échec et mat" if board.is_checkmate() else "Nulle" if h["Result"] == "1/2-1/2" else "Abandon"
    mat = {c: count_material(board, c) for c in (chess.WHITE, chess.BLACK)}

    rows = []
    for c in (chess.WHITE, chess.BLACK):
        s = stats[c]
        n_m, n_e = max(1, (ply + int(c)) // 2), max(1, len(s["cpls"]))
        rows.append({
            "game_id": game_id, "player_name": h.get("White" if c else "Black", "Unknown"),
            "base_time": base_t, "increment": inc, "time_category": cat,
            "result_score": RES_MAP[h["Result"]][0 if c else 1], "termination_type": term,
            "eco_code": h.get("ECO", "Unknown"), "total_plies": ply, "mat_end_diff": mat[c] - mat[not c],
            "did_castle": s["did_castle"], "castle_ply": s["castle_ply"],
            "acpl": mean_nan(s["cpls"]), "acpl_opening": mean_nan(s["open"]),
            "acpl_middlegame": mean_nan(s["mid"]), "acpl_endgame": mean_nan(s["end"]),
            "avg_win_pct_loss": mean_nan(s["win"]), "accuracy_pct": mean_nan(s["accs"]),
            "blunder_rate": s["blunders"] / n_e, "mistake_rate": s["mistakes"] / n_e,
            "inaccuracy_rate": s["inacc"] / n_e,
            "eval_volatility": float(np.std(s["evals"])) if len(s["evals"]) > 1 else np.nan,
            "avg_time_per_move": mean_nan(s["times"]),
            "std_time_per_move": float(np.std(s["times"])) if s["times"] else np.nan,
            "opening_time_avg": mean_nan(s["open_t"]),
            "time_trouble_rate": (s["trouble"] / n_m) if s["times"] else np.nan,
            "final_clock_ratio": (s["last_clk"] / base_t) if s["times"] and base_t > 0 else np.nan,
        })
    return rows


def aggregate_player_profiles(df_games: pd.DataFrame) -> pd.DataFrame:
    num_cols = [
        "base_time", "increment", "result_score", "total_plies", "mat_end_diff",
        "did_castle", "castle_ply", "acpl", "acpl_opening", "acpl_middlegame",
        "acpl_endgame", "avg_win_pct_loss", "accuracy_pct", "blunder_rate",
        "mistake_rate", "inaccuracy_rate", "eval_volatility", "avg_time_per_move",
        "std_time_per_move", "opening_time_avg", "time_trouble_rate", "final_clock_ratio",
    ]
    grp = df_games.groupby(["player_name", "time_category"], sort=False)
    agg_df = grp[num_cols].mean().add_suffix("_mean")
    agg_df["acpl_median"] = grp["acpl"].median()
    agg_df = agg_df.join(grp[KEY_STD_COLS].std().add_suffix("_std"))
    term_rates = (
        pd.crosstab([df_games["player_name"], df_games["time_category"]], df_games["termination_type"], normalize="index")
        .reindex(columns=TERM_CATEGORIES, fill_value=0.0)
        .add_prefix("term_rate_")
    )
    agg_df = agg_df.join(term_rates)
    agg_df["n_games"] = grp.size()
    agg_df["top_eco_code"] = grp["eco_code"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown")
    agg_df["eco_diversity"] = grp["eco_code"].nunique() / agg_df["n_games"]
    return agg_df.reset_index()


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


def analyze_pgn(pgn_text: str, stockfish_path: str, time_limit: float = 0.05):
    pipeline = load_pipeline()
    feature_cols = list(pipeline.feature_names_in_)

    # Lecture rapide de toutes les parties pour afficher la progression
    stream = io.StringIO(pgn_text)
    games = []
    while (g := chess.pgn.read_game(stream)) is not None:
        games.append(g)
    if not games:
        raise ValueError("Aucune partie trouvée dans le PGN.")

    all_rows, tc_labels, skipped = [], [], []
    engine = None
    progress = st.progress(0, text="Analyse des parties en cours...")

    try:
        for idx, game in enumerate(games, 1):
            h = game.headers
            plies = sum(1 for _ in game.mainline_moves())
            if (
                plies < MIN_PLIES
                or h.get("Variant", "Standard") != "Standard"
                or h.get("Result") not in RES_MAP
                or parse_time_control(h.get("TimeControl", ""))[3] == "Unknown"
            ):
                skipped.append(f"Partie #{idx} ignorée (moins de 10 coups ou cadence non reconnue).")
                progress.progress(idx / len(games))
                continue

            if game.next().eval() is None:
                if not stockfish_path or not os.path.exists(stockfish_path):
                    raise ValueError("PGN sans évaluations [%eval] : indiquez un chemin Stockfish valide dans la barre latérale.")
                if engine is None:
                    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                board = game.board()
                for node in game.mainline():
                    board.push(node.move)
                    if not board.is_game_over():
                        info = engine.analyse(board, chess.engine.Limit(time=time_limit))
                        if "score" in info:
                            node.set_eval(info["score"])

            rows = extract_features_from_game(game, game_id=idx)
            if rows:
                all_rows.extend(rows)
                tc_labels.extend([h.get("TimeControl", "?")] * 2)
            else:
                skipped.append(f"Partie #{idx} ignorée (évaluations insuffisantes).")
            progress.progress(idx / len(games), text=f"Partie {idx}/{len(games)} analysée")
    finally:
        if engine is not None:
            engine.quit()
        progress.empty()

    if not all_rows:
        raise ValueError("Aucune partie valide exploitable dans ce PGN.")

    df_games = pd.DataFrame(all_rows)
    df_games["tc_raw"] = tc_labels
    df_profiles = aggregate_player_profiles(df_games)
    df_profiles["Elo prédit"] = np.round(pipeline.predict(df_profiles[feature_cols])).astype(int)
    return df_games, df_profiles, skipped, pipeline, feature_cols


DEMO_PGN = """
[Event "Rated Rapid game"]
[White "Alex"]
[Black "Adversaire 1"]
[Result "1-0"]
[TimeControl "600+0"]
[ECO "C50"]
[Termination "Normal"]

1. e4 { [%eval 0.20] [%clk 0:10:00] } 1... e5 { [%eval 0.22] [%clk 0:10:00] }
2. Nf3 { [%eval 0.25] [%clk 0:09:58] } 2... Nc6 { [%eval 0.21] [%clk 0:09:57] }
3. Bc4 { [%eval 0.15] [%clk 0:09:55] } 3... Bc5 { [%eval 0.18] [%clk 0:09:54] }
4. O-O { [%eval 0.10] [%clk 0:09:50] } 4... Nf6 { [%eval 0.12] [%clk 0:09:49] }
5. d3 { [%eval 0.05] [%clk 0:09:45] } 5... d6 { [%eval 0.08] [%clk 0:09:44] }
6. c3 { [%eval 0.04] [%clk 0:09:40] } 6... O-O { [%eval 0.07] [%clk 0:09:38] }
7. Bb3 { [%eval 0.02] [%clk 0:09:32] } 7... a6 { [%eval 0.05] [%clk 0:09:30] }
8. Nbd2 { [%eval 0.00] [%clk 0:09:25] } 8... Ba7 { [%eval 0.02] [%clk 0:09:20] }
9. h3 { [%eval 0.00] [%clk 0:09:15] } 9... h6 { [%eval 0.00] [%clk 0:09:10] }
10. Re1 { [%eval 0.05] [%clk 0:09:05] } 10... Re8 { [%eval 0.05] [%clk 0:09:00] }
11. Nf1 { [%eval 0.10] [%clk 0:08:50] } 11... Be6 { [%eval 0.15] [%clk 0:08:45] }
12. Ng3 { [%eval 0.12] [%clk 0:08:40] } 12... Qd7 { [%eval 1.85] [%clk 0:08:30] }
13. Bxe6 { [%eval 1.80] [%clk 0:08:20] } 13... fxe6 { [%eval 2.10] [%clk 0:08:22] }
14. d4 { [%eval 2.20] [%clk 0:08:10] } 14... exd4 { [%eval 4.50] [%clk 0:08:15] }
15. cxd4 { [%eval 4.40] [%clk 0:08:05] } 15... e5 { [%eval 5.10] [%clk 0:08:08] } 1-0

[Event "Rated Rapid game"]
[White "Adversaire 2"]
[Black "Alex"]
[Result "1/2-1/2"]
[TimeControl "600+0"]
[ECO "D63"]
[Termination "Normal"]

1. d4 { [%eval 0.18] [%clk 0:10:00] } 1... d5 { [%eval 0.20] [%clk 0:10:00] }
2. c4 { [%eval 0.22] [%clk 0:09:57] } 2... e6 { [%eval 0.25] [%clk 0:09:56] }
3. Nc3 { [%eval 0.20] [%clk 0:09:54] } 3... Nf6 { [%eval 0.24] [%clk 0:09:52] }
4. Bg5 { [%eval 0.18] [%clk 0:09:49] } 4... Be7 { [%eval 0.20] [%clk 0:09:48] }
5. e3 { [%eval 0.15] [%clk 0:09:45] } 5... O-O { [%eval 0.19] [%clk 0:09:44] }
6. Nf3 { [%eval 0.21] [%clk 0:09:40] } 6... Nbd7 { [%eval 0.23] [%clk 0:09:38] }
7. Rc1 { [%eval 0.19] [%clk 0:09:32] } 7... c6 { [%eval 0.22] [%clk 0:09:31] }
8. Bd3 { [%eval 0.15] [%clk 0:09:24] } 8... dxc4 { [%eval 0.20] [%clk 0:09:20] }
9. Bxc4 { [%eval 0.18] [%clk 0:09:18] } 9... Nd5 { [%eval 0.24] [%clk 0:09:12] }
10. Bxe7 { [%eval 0.16] [%clk 0:09:05] } 10... Qxe7 { [%eval 0.18] [%clk 0:09:07] }
11. O-O { [%eval 0.12] [%clk 0:08:55] } 11... Nxc3 { [%eval 0.15] [%clk 0:08:58] }
12. Rxc3 { [%eval 0.14] [%clk 0:08:48] } 12... e5 { [%eval 0.16] [%clk 0:08:45] }
13. dxe5 { [%eval 0.10] [%clk 0:08:35] } 13... Nxe5 { [%eval 0.12] [%clk 0:08:38] }
14. Nxe5 { [%eval 0.08] [%clk 0:08:25] } 14... Qxe5 { [%eval 0.05] [%clk 0:08:30] } 1/2-1/2

[Event "Rated Blitz game"]
[White "Alex"]
[Black "Adversaire 3"]
[Result "1-0"]
[TimeControl "180+2"]
[ECO "B90"]
[Termination "Normal"]

1. e4 { [%eval 0.20] [%clk 0:03:00] } 1... c5 { [%eval 0.25] [%clk 0:03:00] }
2. Nf3 { [%eval 0.22] [%clk 0:02:59] } 2... d6 { [%eval 0.28] [%clk 0:02:59] }
3. d4 { [%eval 0.30] [%clk 0:02:58] } 3... cxd4 { [%eval 0.32] [%clk 0:02:58] }
4. Nxd4 { [%eval 0.28] [%clk 0:02:58] } 4... Nf6 { [%eval 0.30] [%clk 0:02:57] }
5. Nc3 { [%eval 0.26] [%clk 0:02:57] } 5... a6 { [%eval 0.35] [%clk 0:02:56] }
6. Be3 { [%eval 0.30] [%clk 0:02:55] } 6... e5 { [%eval 0.38] [%clk 0:02:54] }
7. Nb3 { [%eval 0.28] [%clk 0:02:52] } 7... Be6 { [%eval 0.32] [%clk 0:02:51] }
8. f3 { [%eval 0.25] [%clk 0:02:49] } 8... Be7 { [%eval 0.35] [%clk 0:02:48] }
9. Qd2 { [%eval 0.30] [%clk 0:02:45] } 9... O-O { [%eval 0.40] [%clk 0:02:43] }
10. O-O-O { [%eval 0.32] [%clk 0:02:40] } 10... Nbd7 { [%eval 0.45] [%clk 0:02:38] }
11. g4 { [%eval 0.42] [%clk 0:02:34] } 11... b5 { [%eval 0.80] [%clk 0:02:30] }
12. g5 { [%eval 0.78] [%clk 0:02:28] } 12... Nh5 { [%eval 1.45] [%clk 0:02:22] } 1-0
"""

def render_hd_plot(fig, dpi: int = 300):
    """Affiche une figure Matplotlib/SHAP en haute résolution (style Retina) dans Streamlit."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    st.image(buf, use_container_width=True)

# --- Interface Streamlit ---
st.title("♟️ Prédiction du classement Elo aux échecs")
st.markdown(
    "Modèle CatBoost entraîné sur 120 000 parties Lichess agrégées.  \n"
    "Compatible Lichess et Chess.com (les prédictions sont exprimées en équivalent Elo Lichess)."
)

STOCKFISH_BIN = (
    os.getenv("STOCKFISH_PATH")
    or shutil.which("stockfish")
    or ("/usr/games/stockfish" if os.path.exists("/usr/games/stockfish") else None)
)

with st.sidebar:
    st.header("Paramètres")
    sf_time = st.slider("Temps Stockfish par coup (s)", 0.02, 0.20, 0.05, 0.01)
    if STOCKFISH_BIN and os.path.exists(STOCKFISH_BIN):
        sf_path = STOCKFISH_BIN
        st.success("Moteur Stockfish prêt.")
    else:
        sf_path = st.text_input("Chemin local vers Stockfish (si PGN sans [%eval])", value="")
    st.info("Pour une prédiction optimale, fournissez 5 à 10 parties d'un même joueur dans la même cadence.")

tab_upload, tab_paste = st.tabs(["Importer un fichier .pgn", "Coller un texte PGN"])
pgn_input = ""
with tab_upload:
    uploaded_files = st.file_uploader(
        "Choisissez un ou plusieurs fichiers PGN",
        type=["pgn", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        pgn_input = "\n\n".join(
            f.getvalue().decode("utf-8", errors="replace") for f in uploaded_files
        )
with tab_paste:
    pasted = st.text_area("Collez le contenu PGN ici", height=200)
    if pasted.strip():
        pgn_input = pasted

col_run, col_demo = st.columns([2, 1])
with col_run:
    run_clicked = st.button("Prédire le classement Elo", type="primary", use_container_width=True, disabled=not pgn_input.strip())
with col_demo:
    demo_clicked = st.button("Tester avec un exemple", use_container_width=True)

if run_clicked or demo_clicked:
    source_pgn = DEMO_PGN if demo_clicked else pgn_input
    try:
        st.session_state["results"] = analyze_pgn(source_pgn, sf_path, sf_time)
    except Exception as e:
        st.error(str(e))

if "results" in st.session_state:
    df_games, df_profiles, skipped, pipeline, feature_cols = st.session_state["results"]

    for msg in skipped:
        st.warning(msg)

    # 1. Synthèse par cadence
    st.subheader("Classement Elo prédit par joueur et par cadence")
    present_cats = [c for c in CADENCE_ORDER if c in df_profiles["time_category"].values]
    cols = st.columns(min(len(present_cats), 2))

    for i, cat in enumerate(present_cats):
        sub = (
            df_profiles[df_profiles["time_category"] == cat][
                ["player_name", "n_games", "accuracy_pct_mean", "acpl_mean", "Elo prédit"]
            ]
            .rename(columns={
                "player_name": "Joueur", "n_games": "Parties",
                "accuracy_pct_mean": "Précision moy. (%)", "acpl_mean": "ACPL moyen",
            })
            .sort_values(["Elo prédit", "Parties"], ascending=[False, False])
        )
        with cols[i % len(cols)]:
            st.markdown(f"**{cat}**")
            st.dataframe(
                sub.round({"Précision moy. (%)": 1, "ACPL moyen": 1}),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Précision moy. (%)": st.column_config.NumberColumn(format="%.1f"),
                    "ACPL moyen": st.column_config.NumberColumn(format="%.1f"),
                },
            )

    # 2. Détail des parties
    n_games = df_games["game_id"].nunique()
    expander_title = (
        "Voir le détail de la partie analysée"
        if n_games == 1
        else f"Voir le détail des {n_games} parties analysées"
    )

    with st.expander(expander_title):
        grouped_games = list(df_games.groupby("game_id", sort=False))
        n_cols = 2

        for start in range(0, len(grouped_games), n_cols):
            cols = st.columns(n_cols)
            for col, (game_id, grp) in zip(cols, grouped_games[start : start + n_cols]):
                cadence = f"{grp['time_category'].iloc[0]} ({grp['tc_raw'].iloc[0]})"
                tbl = pd.DataFrame({
                    "Camp": ["Blancs", "Noirs"][:len(grp)],
                    "Joueur": grp["player_name"].values,
                    "Résultat": grp["result_score"].map(RESULT_LABELS).values,
                    "Précision (%)": grp["accuracy_pct"].round(1).values,
                    "ACPL": grp["acpl"].round(1).values,
                })
                with col:
                    st.markdown(f"**Partie #{game_id} – {cadence}**")
                    st.dataframe(
                        tbl,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Précision (%)": st.column_config.NumberColumn(format="%.1f"),
                            "ACPL": st.column_config.NumberColumn(format="%.1f"),
                        },
                    )

    # 3. Analyse détaillée du profil sélectionné
    st.subheader("Analyse détaillée du profil sélectionné")
    options = [
        f"{r['player_name']} – {r['time_category']} ({r['n_games']} {'partie' if r['n_games'] == 1 else 'parties'} | Elo prédit : {r['Elo prédit']})"
        for _, r in df_profiles.iterrows()
    ]
    selected_idx = st.selectbox("Choisir un profil à expliquer :", range(len(options)), format_func=lambda i: options[i])


    row_sel = df_profiles.iloc[selected_idx]
    col_g1, col_g2 = st.columns(2)

    # Graphique 1 : ACPL par phase de jeu
    with col_g1:
        phases = {
            "Ouverture\n(coups 1–10)": row_sel["acpl_opening_mean"],
            "Milieu de jeu\n(coups 11–30)": row_sel["acpl_middlegame_mean"],
            "Finale\n(coups 31+)": row_sel["acpl_endgame_mean"],
        }
        valid_phases = {k: v for k, v in phases.items() if pd.notna(v)}
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        bars = ax.bar(valid_phases.keys(), valid_phases.values(), color=["#4C72B0", "#DD8452", "#55A868"])
        ax.bar_label(bars, fmt="%.1f", padding=3)
        ax.set(ylabel="ACPL moyen", title="ACPL moyen par phase de jeu (plus bas = meilleur)")
        ax.set_ylim(0, max(valid_phases.values(), default=50) * 1.2)
        plt.tight_layout()
        render_hd_plot(fig)

    # Graphique 2 : Évolution de la précision partie par partie (si >= 2 parties)
    with col_g2:
        player_games = df_games[
            (df_games["player_name"] == row_sel["player_name"])
            & (df_games["time_category"] == row_sel["time_category"])
        ].reset_index(drop=True)

        if len(player_games) >= 2:
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            x_idx = range(1, len(player_games) + 1)
            ax.plot(x_idx, player_games["accuracy_pct"], marker="o", lw=2, color="#4C72B0")
            ax.axhline(player_games["accuracy_pct"].mean(), color="red", ls="--", lw=1.2, label="Moyenne")
            ax.set(
                xticks=list(x_idx), ylim=(0, 105),
                xlabel="Numéro de partie", ylabel="Précision (%)",
                title=f"Évolution de la précision ({row_sel['player_name']})",
            )
            ax.legend()
            plt.tight_layout()
            render_hd_plot(fig)
        else:
            st.info("Importez au moins 2 parties de ce joueur dans cette cadence pour afficher l'évolution de sa précision.")

    # Graphique 3 : SHAP
    prep, model = pipeline["prep"], pipeline["model"]
    X_one = df_profiles.iloc[[selected_idx]][feature_cols]
    X_trans = pd.DataFrame(prep.transform(X_one), columns=[c.split("__")[-1] for c in prep.get_feature_names_out()])
    shap_vals = shap.TreeExplainer(model)(X_trans)

    fig = plt.figure(figsize=(8, 4.5))
    shap.plots.waterfall(shap_vals[0], max_display=10, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)

    FEATURE_GLOSSARY = [
        ("accuracy_pct_mean", "Précision moyenne des coups en % sur les parties (formule Lichess, de 0 à 100 %)."),
        ("accuracy_pct_std", "Écart-type de la précision (%) d'une partie à l'autre (mesure la régularité)."),
        ("acpl_endgame_mean", "Perte moyenne en centipions (ACPL) en finale (à partir du 31e coup)."),
        ("acpl_mean", "Perte moyenne en centipions par coup (plus c'est bas, plus le jeu est précis)."),
        ("acpl_median", "Médiane de l'ACPL sur les parties du joueur (moins sensible aux parties atypiques)."),
        ("acpl_middlegame_mean", "Perte moyenne en centipions (ACPL) en milieu de jeu (coups 11 à 30)."),
        ("acpl_opening_mean", "Perte moyenne en centipions (ACPL) dans l'ouverture (10 premiers coups)."),
        ("acpl_opening_std", "Écart-type de l'ACPL d'ouverture d'une partie à l'autre."),
        ("acpl_std", "Écart-type de l'ACPL global d'une partie à l'autre (mesure la constance du niveau)."),
        ("avg_time_per_move_mean", "Temps moyen passé par coup (en secondes)."),
        ("avg_time_per_move_std", "Écart-type du temps moyen par coup d'une partie à l'autre."),
        ("avg_win_pct_loss_mean", "Perte moyenne de probabilité de victoire par coup (en points de %)."),
        ("base_time_mean", "Temps initial moyen de la cadence à la pendule (en secondes)."),
        ("blunder_rate_mean", "Proportion moyenne de gaffes (perte ≥ 200 centipions sur un coup)."),
        ("blunder_rate_std", "Écart-type du taux de gaffes d'une partie à l'autre."),
        ("castle_ply_mean", "Demi-coup moyen auquel le joueur effectue son roque."),
        ("did_castle_mean", "Proportion des parties dans lesquelles le joueur a roqué (entre 0 et 1)."),
        ("eco_diversity", "Diversité du répertoire d'ouvertures (nombre de codes ECO distincts / nombre de parties)."),
        ("eval_volatility_mean", "Écart-type moyen de l'évaluation au cours d'une partie (mesure son caractère chaotique)."),
        ("final_clock_ratio_mean", "Part moyenne du temps initial encore restante à la pendule en fin de partie."),
        ("inaccuracy_rate_mean", "Proportion moyenne d'imprécisions (perte entre 50 et 100 centipions)."),
        ("increment_mean", "Incrément moyen ajouté par coup selon la cadence (en secondes)."),
        ("mat_end_diff_mean", "Différentiel matériel moyen en fin de partie (positif = avantage matériel)."),
        ("mistake_rate_mean", "Proportion moyenne d'erreurs (perte entre 100 et 200 centipions)."),
        ("opening_time_avg_mean", "Temps moyen par coup dans l'ouverture (un temps faible reflète la maîtrise théorique)."),
        ("result_score_mean", "Score moyen sur les parties (1 = victoire, 0,5 = nulle, 0 = défaite)."),
        ("std_time_per_move_mean", "Variabilité moyenne du temps de réflexion au sein d'une partie (coups rapides vs réfléchis)."),
        ("term_rate_Abandon", "Proportion des parties terminées par abandon."),
        ("term_rate_Échec et mat", "Proportion des parties terminées par échec et mat."),
        ("term_rate_Nulle", "Proportion des parties terminées par match nul."),
        ("term_rate_Temps écoulé", "Proportion des parties terminées par perte au temps."),
        ("time_category_Blitz", "Indicateur binaire (1 si le profil est en cadence Blitz, 0 sinon)."),
        ("time_category_Bullet", "Indicateur binaire (1 si le profil est en cadence Bullet, 0 sinon)."),
        ("time_category_Classique", "Indicateur binaire (1 si le profil est en cadence Classique, 0 sinon)."),
        ("time_category_Rapide", "Indicateur binaire (1 si le profil est en cadence Rapide, 0 sinon)."),
        ("time_trouble_rate_mean", "Proportion moyenne de coups joués en zeitnot (< 10 s ou < 10 % du temps initial)."),
        ("top_eco_code", "Code ECO de l'ouverture la plus jouée (encodé selon l'Elo moyen associé)."),
        ("total_plies_mean", "Longueur moyenne des parties en demi-coups (1 coup complet = 2 demi-coups)."),
        ("total_plies_std", "Écart-type de la longueur des parties en demi-coups."),
    ]

    with st.expander("Comprendre les variables du graphique SHAP"):
        st.caption("Chaque barre rouge ajoute des points Elo à l'estimation moyenne de référence, chaque barre bleue en retire.")
        st.dataframe(
            pd.DataFrame(FEATURE_GLOSSARY, columns=["Variable", "Signification"]),
            hide_index=True,
            use_container_width=True,
            height=420,
        )