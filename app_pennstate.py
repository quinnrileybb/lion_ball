import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec
import numpy as np
import io
import zipfile
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.utils import ImageReader

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Penn State Baseball – Hitter Report",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.main .block-container {
    padding-top: 1rem; padding-bottom: 1rem;
    padding-left: 1.5rem; padding-right: 1.5rem;
    max-width: none;
}
</style>
""", unsafe_allow_html=True)

# ── Strike-zone geometry (feet) ────────────────────────────────────────────────
RULEBOOK_LEFT   = -0.83083
RULEBOOK_RIGHT  =  0.83083
RULEBOOK_BOTTOM =  1.5
RULEBOOK_TOP    =  3.3775

SZ_W = RULEBOOK_RIGHT - RULEBOOK_LEFT
SZ_H = RULEBOOK_TOP - RULEBOOK_BOTTOM

SHADOW_MULT   = 1.33
SHD_EXT_W     = (SZ_W * SHADOW_MULT - SZ_W) / 2
SHD_EXT_H     = (SZ_H * SHADOW_MULT - SZ_H) / 2
SHADOW_LEFT   = RULEBOOK_LEFT  - SHD_EXT_W
SHADOW_RIGHT  = RULEBOOK_RIGHT + SHD_EXT_W
SHADOW_BOTTOM = RULEBOOK_BOTTOM - SHD_EXT_H
SHADOW_TOP    = RULEBOOK_TOP    + SHD_EXT_H

# ── Color / marker palettes ────────────────────────────────────────────────────
PITCH_CALL_PALETTE = {
    'StrikeCalled':        'orange',
    'BallCalled':          'green',
    'BallinDirt':          'green',
    'Foul':                'tan',
    'InPlay':              'blue',
    'FoulBallNotFieldable':'tan',
    'StrikeSwinging':      'red',
    'BallIntentional':     'green',
    'FoulBallFieldable':   'tan',
    'HitByPitch':          'lime',
}

PITCH_CALL_LEGEND = {
    'StrikeCalled':        'Strike Called',
    'BallCalled':          'Ball',
    'BallinDirt':          'Ball',
    'Foul':                'Foul',
    'InPlay':              'In Play',
    'FoulBallNotFieldable':'Foul',
    'StrikeSwinging':      'Strike Swinging',
    'BallIntentional':     'Ball',
    'FoulBallFieldable':   'Foul',
    'HitByPitch':          'Hit By Pitch',
}

PITCH_TYPE_MARKERS = {
    'Fastball':  'o',
    'Curveball': 's',
    'Slider':    '^',
    'Changeup':  'D',
}

PLAY_RESULT_STYLES = {
    "Single":  ("blue",   "o"),
    "Double":  ("purple", "o"),
    "Triple":  ("gold",   "o"),
    "HomeRun": ("orange", "o"),
    "Out":     ("black",  "o"),
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file, low_memory=False)
    # Filter to Penn State batters
    if 'BatterTeam' in df.columns:
        df = df[df['BatterTeam'] == 'PEN_NIT']
    if 'AutoPitchType' in df.columns:
        df['AutoPitchType'] = df['AutoPitchType'].str.strip().str.capitalize()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    return df


def build_game_options(df):
    options = {}
    has_game_num = "game_num" in df.columns and df["game_num"].notna().any()
    has_uid      = "GameUID" in df.columns
    entries = []

    if has_game_num:
        id_cols = ["Date", "game_num"]
        if has_uid:       id_cols.append("GameUID")
        if "PitcherTeam" in df.columns: id_cols.append("PitcherTeam")
        reg = (
            df[df["game_num"].notna()]
            .drop_duplicates(subset=(["GameUID"] if has_uid else ["Date"]))
            [id_cols]
            .sort_values("game_num")
        )
        for _, row in reg.iterrows():
            opp   = row["PitcherTeam"] if "PitcherTeam" in row.index else "OPP"
            label = f"Game {int(row['game_num'])} vs {opp} ({row['Date']})"
            uid   = row["GameUID"] if has_uid else None
            entries.append((row["Date"], label, (row["Date"], uid)))

    reg_dates = {e[0] for e in entries}
    for d in df["Date"].unique():
        if d not in reg_dates:
            entries.append((d, d, (d, None)))
    entries.sort(key=lambda x: x[0])
    for _, label, value in entries:
        options[label] = value
    return options


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf.read()


def build_hitter_figures(batter_df, batter_name, game_label, logo_img=None):
    """Return (pitch_plot_fig, batted_ball_fig) for one hitter."""

    # ── Pitch-by-pitch figure ─────────────────────────────────────────────────
    plate_appearance_groups = list(batter_df.groupby((batter_df['PitchofPA'] == 1).cumsum()))
    num_pa = len(plate_appearance_groups)

    fig_width, fig_height = 20, 11
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = GridSpec(3, 4, figure=fig,
                   width_ratios=[1.5, 1.5, 1.5, 1.2],
                   height_ratios=[1, 1, 1])
    gs.update(wspace=0.25, hspace=0.35)

    axes = []
    for i in range(min(num_pa, 9)):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(1, 4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1)
        axes.append(ax)

    SZ_W_HALF = (17 / 12) / 2
    sz_params   = {'x_start': -SZ_W_HALF,          'y_start': 1.5,  'width': 17/12,      'height': 3.3775 - 1.5}
    heart_params= {'x_start': -SZ_W_HALF + (17/12)*0.33, 'y_start': 1.5 + (3.3775-1.5)*0.33,
                   'width': (17/12)*0.34, 'height': (3.3775-1.5)*0.34}
    shadow_params={'x_start': -SZ_W_HALF - 0.2,    'y_start': 1.3,  'width': 17/12+0.4, 'height': 3.6-1.3}

    table_data = []

    for i, (pa_id, pa_data) in enumerate(plate_appearance_groups, start=1):
        if i > 9:
            break
        ax = axes[i - 1]
        pitcher_throws = pa_data.iloc[0]['PitcherThrows']
        handedness     = 'RHP' if pitcher_throws == 'Right' else 'LHP'
        pitcher_name   = pa_data.iloc[0]['Pitcher']
        ax.set_title(f'PA {i} vs {handedness}', fontsize=18, fontweight='bold')
        ax.text(0.5, -0.12, f'P: {pitcher_name}', fontsize=12, fontstyle='italic',
                ha='center', transform=ax.transAxes)

        # Zones
        ax.add_patch(plt.Rectangle((shadow_params['x_start'], shadow_params['y_start']),
                                   shadow_params['width'], shadow_params['height'],
                                   fill=False, color='gray', linestyle='--', linewidth=2))
        ax.add_patch(plt.Rectangle((sz_params['x_start'], sz_params['y_start']),
                                   sz_params['width'], sz_params['height'],
                                   fill=False, color='black', linewidth=2))
        ax.add_patch(plt.Rectangle((heart_params['x_start'], heart_params['y_start']),
                                   heart_params['width'], heart_params['height'],
                                   fill=False, color='red', linestyle='--', linewidth=2))

        pa_rows = []
        for _, row in pa_data.iterrows():
            sns.scatterplot(
                x=[row['PlateLocSide']], y=[row['PlateLocHeight']],
                hue=[row['PitchCall']], palette=PITCH_CALL_PALETTE,
                marker=PITCH_TYPE_MARKERS.get(row['AutoPitchType'], 'o'),
                s=200, legend=False, ax=ax
            )
            offset = -0.05 if row['AutoPitchType'] == 'Slider' else 0
            ax.text(row['PlateLocSide'], row['PlateLocHeight'] + offset,
                    f"{int(row['PitchofPA'])}", color='white', fontsize=10,
                    ha='center', va='center', weight='bold')

            pitch_speed = f"{round(row['RelSpeed'], 1)} MPH"
            pitch_type  = row['AutoPitchType']
            if row.name == pa_data.index[-1]:
                play_result  = row['PlayResult']
                kor_bb       = row['KorBB']
                pitch_call   = row['PitchCall']
                outcome_x    = next((r for r in [play_result, kor_bb, pitch_call] if r != "Undefined"), "Undefined")
            else:
                outcome_x = row['PitchCall']
            pa_rows.append([f"Pitch {int(row['PitchofPA'])}", f"{pitch_speed} {pitch_type}", outcome_x])

        table_data.append([f'PA {i}', '', ''])
        table_data.extend(pa_rows)

    # Legends
    unique_calls = {}
    for pc, color in PITCH_CALL_PALETTE.items():
        lbl = PITCH_CALL_LEGEND.get(pc, pc)
        if lbl not in unique_calls:
            unique_calls[lbl] = color

    handles1 = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=c,
                            markersize=8, linestyle='', label=l)
                for l, c in unique_calls.items()]
    handles2 = [plt.Line2D([0],[0], marker=m, color='black', markersize=8,
                            linestyle='', label=l)
                for l, m in PITCH_TYPE_MARKERS.items()]

    fig.legend(handles=handles1, title='Pitch Call (Colors)',
               loc='lower center', bbox_to_anchor=(0.5, 0.02),
               ncol=len(unique_calls), fontsize=12, title_fontsize=14, frameon=False)
    fig.legend(handles=handles2, title='Pitch Type (Shapes)',
               loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=len(PITCH_TYPE_MARKERS), fontsize=12, title_fontsize=14, frameon=False)
    fig.subplots_adjust(bottom=0.15)

    # Pitch-by-pitch table column
    ax_table = fig.add_subplot(gs[:, 3])
    ax_table.axis('off')
    y_pos = 1.0
    for row in table_data:
        if 'PA' in row[0]:
            ax_table.text(0.05, y_pos, row[0], fontsize=14, fontweight='bold', fontstyle='italic')
            ax_table.axhline(y=y_pos - 0.01, color='black', linewidth=1)
            y_pos -= 0.05
        else:
            ax_table.text(0.05, y_pos, f"  {row[0]}  |  {row[1]}  |  {row[2]}", fontsize=12)
            y_pos -= 0.04

    # Stats
    whiffs     = batter_df['PitchCall'].eq('StrikeSwinging').sum()
    hard_hits  = batter_df[(batter_df['PitchCall']=='InPlay') & (batter_df['ExitSpeed']>=95)].shape[0]
    barrels    = batter_df[(batter_df['ExitSpeed']>=95) & (batter_df['Angle'].between(10,35))].shape[0]
    swing_calls= ['Foul','InPlay','StrikeSwinging','FoulBallFieldable','FoulBallNotFieldable']
    swings     = batter_df[batter_df['PitchCall'].isin(swing_calls)]
    chase      = swings[
        (swings['PlateLocSide'] < -0.7083) | (swings['PlateLocSide'] > 0.7083) |
        (swings['PlateLocHeight'] < 1.5)   | (swings['PlateLocHeight'] > 3.3775)
    ].shape[0]

    fig.suptitle(f"{batter_name} Report — {game_label}", fontsize=24, weight='bold')
    fig.text(0.5, 0.93,
             f"Whiffs: {whiffs}    Hard Hit: {hard_hits}    Barrels: {barrels}    Chase: {chase}",
             fontsize=16, ha='center')

    if logo_img is not None:
        logo_ax = fig.add_axes([0.85, 0.92, 0.12, 0.12])
        logo_ax.imshow(logo_img)
        logo_ax.axis('off')

    # ── Batted ball figure ────────────────────────────────────────────────────
    bb_fig, bb_ax = plt.subplots(figsize=(6, 6))

    LF, LC, CF_dist, RC, RF = 330, 365, 390, 365, 330
    angles    = np.linspace(-45, 45, 500)
    distances = np.interp(angles, [-45,-30,0,30,45], [LF,LC,CF_dist,RC,RF])
    x_of = distances * np.sin(np.radians(angles))
    y_of = distances * np.cos(np.radians(angles))
    bb_ax.plot(x_of, y_of, color='black', linewidth=2)

    foul_xl = [-LF*np.sin(np.radians(45)), 0];  foul_yl = [LF*np.cos(np.radians(45)), 0]
    foul_xr = [ RF*np.sin(np.radians(45)), 0];  foul_yr = [RF*np.cos(np.radians(45)), 0]
    bb_ax.plot(foul_xl, foul_yl, color='black')
    bb_ax.plot(foul_xr, foul_yr, color='black')

    bases_x = [0, 90, 0, -90, 0]
    bases_y = [0, 90, 180, 90, 0]
    bb_ax.plot(bases_x, bases_y, color='brown', linewidth=2)

    in_play = batter_df[batter_df['PitchCall'] == 'InPlay'].copy()
    for pa_number, pa_data in plate_appearance_groups:
        if pa_data.empty:
            continue
        last = pa_data.iloc[-1]
        if last['PitchCall'] != 'InPlay':
            continue
        bearing  = np.radians(last['Bearing'])
        distance = last['Distance']
        ev       = round(last['ExitSpeed'], 1) if pd.notnull(last.get('ExitSpeed')) else 'NA'
        result   = last['PlayResult']
        x = distance * np.sin(bearing)
        y = distance * np.cos(bearing)
        color, marker = PLAY_RESULT_STYLES.get(result, ('black','o'))
        bb_ax.scatter(x, y, color=color, marker=marker, s=200, edgecolor='black')
        bb_ax.text(x, y, str(pa_number), color='white', fontsize=12,
                   fontweight='bold', ha='center', va='center')
        bb_ax.text(x, y - 15, f"{ev} mph" if ev != 'NA' else 'NA',
                   color='red', fontsize=10, fontweight='bold', ha='center')

    bb_ax.set_xticks([]); bb_ax.set_yticks([])
    bb_ax.axis('equal')
    bb_ax.set_title(f"Batted Ball Locations — {batter_name}", fontsize=16)

    legend_els = [
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='blue',   markersize=10, label='Single'),
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Double'),
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='gold',   markersize=10, label='Triple'),
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='HomeRun'),
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='black',  markersize=10, label='Out'),
    ]
    bb_fig.legend(handles=legend_els, loc='lower center', ncol=5, fontsize=9, frameon=False)
    plt.subplots_adjust(bottom=0.15)

    return fig, bb_fig


def build_pdf_for_batter(pitch_fig, bb_fig, batter_name, game_label):
    """Render both figures into a 2-page landscape PDF, return bytes."""
    pdf_buf = io.BytesIO()
    page_w, page_h = landscape(letter)  # 11 x 8.5 in = 792 x 612 pts
    c = rl_canvas.Canvas(pdf_buf, pagesize=(page_w, page_h))

    for fig in [pitch_fig, bb_fig]:
        img_bytes = fig_to_png_bytes(fig)
        img_reader = ImageReader(io.BytesIO(img_bytes))
        c.drawImage(img_reader, 0, 0, width=page_w, height=page_h,
                    preserveAspectRatio=True, anchor='c')
        c.showPage()

    c.save()
    pdf_buf.seek(0)
    return pdf_buf.read()


# ── Streamlit UI ───────────────────────────────────────────────────────────────
st.title("Penn State Baseball — Postgame Hitter Report")

# Logo upload (optional)
with st.sidebar:
    st.header("Settings")
    logo_file = st.file_uploader("Upload Penn State Logo (PNG)", type=['png','jpg','jpeg'])
    logo_img  = None
    if logo_file:
        import matplotlib.image as mpimg
        logo_img = mpimg.imread(logo_file)

# Trackman file upload
uploaded_csv = st.file_uploader(
    "📂 Upload Trackman CSV File",
    type=['csv'],
    help="Upload the Trackman export CSV for the game you want to report on."
)

if not uploaded_csv:
    st.info("Upload a Trackman CSV file above to get started.")
    st.stop()

with st.spinner("Loading data..."):
    data = load_data(uploaded_csv)

if data.empty:
    st.error("No Penn State batter data found in this file. Check that BatterTeam is 'PEN_NIT'.")
    st.stop()

# Game selector
game_options = build_game_options(data)
game_labels  = list(game_options.keys())

col1, col2 = st.columns([3, 1])
with col1:
    selected_label = st.selectbox("Select a Game", options=game_labels, index=len(game_labels)-1)
selected_date, selected_uid = game_options[selected_label]

# Filter to game
filtered_data = data[data["Date"] == selected_date]
if selected_uid is not None and "GameUID" in filtered_data.columns:
    filtered_data = filtered_data[filtered_data["GameUID"] == selected_uid]

unique_batters = sorted(filtered_data['Batter'].dropna().unique())

if not unique_batters:
    st.warning("No batters found for the selected game.")
    st.stop()

st.markdown(f"**{len(unique_batters)} hitters found** for *{selected_label}*")

# ── Preview individual hitter ──────────────────────────────────────────────────
st.markdown("---")
st.subheader("Preview a Hitter")
preview_batter = st.selectbox("Select Hitter to Preview", options=unique_batters)

if preview_batter:
    batter_df = filtered_data[filtered_data['Batter'] == preview_batter]
    if not batter_df.empty:
        with st.spinner(f"Generating report for {preview_batter}..."):
            p_fig, bb_fig = build_hitter_figures(batter_df, preview_batter, selected_label, logo_img)
        st.pyplot(p_fig, use_container_width=True)
        st.pyplot(bb_fig, use_container_width=False)
        plt.close('all')

# ── Export all hitters as PDFs ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("Export PDFs")
st.write("Click below to generate a PDF for **every hitter** from the selected game. "
         "All PDFs will be bundled into a single ZIP file for download.")

if st.button("⬇️  Generate & Download All Hitter PDFs", type="primary"):
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        progress = st.progress(0, text="Building PDFs…")
        for idx, batter in enumerate(unique_batters):
            batter_df = filtered_data[filtered_data['Batter'] == batter]
            if batter_df.empty:
                continue
            p_fig, bb_fig = build_hitter_figures(batter_df, batter, selected_label, logo_img)
            pdf_bytes = build_pdf_for_batter(p_fig, bb_fig, batter, selected_label)
            safe_name = batter.replace(' ', '_').replace('/', '-')
            zf.writestr(f"{safe_name}_report.pdf", pdf_bytes)
            plt.close('all')
            progress.progress((idx + 1) / len(unique_batters),
                              text=f"Built {idx+1}/{len(unique_batters)}: {batter}")

    zip_buf.seek(0)
    game_tag = selected_date.replace('-', '')
    st.download_button(
        label="📦  Download ZIP of PDFs",
        data=zip_buf,
        file_name=f"PSU_Hitters_{game_tag}.zip",
        mime="application/zip",
    )
    st.success(f"✅ {len(unique_batters)} hitter PDFs ready!")
