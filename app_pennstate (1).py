import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.gridspec import GridSpec
import numpy as np
import io
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.utils import ImageReader
import matplotlib.image as mpimg
import os

# ── Penn State brand colors ────────────────────────────────────────────────────
PSU_NAVY  = '#1E407C'
PSU_WHITE = '#FFFFFF'
PSU_LTBLUE = '#96bee6'

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

# ── Palettes ───────────────────────────────────────────────────────────────────
PITCH_CALL_PALETTE = {
    'StrikeCalled':         'orange',
    'BallCalled':           'green',
    'BallinDirt':           'green',
    'Foul':                 'tan',
    'InPlay':               'blue',
    'FoulBallNotFieldable': 'tan',
    'StrikeSwinging':       'red',
    'BallIntentional':      'green',
    'FoulBallFieldable':    'tan',
    'HitByPitch':           'lime',
}
PITCH_CALL_LEGEND = {
    'StrikeCalled':         'Strike Called',
    'BallCalled':           'Ball',
    'BallinDirt':           'Ball',
    'Foul':                 'Foul',
    'InPlay':               'In Play',
    'FoulBallNotFieldable': 'Foul',
    'StrikeSwinging':       'Strike Swinging',
    'BallIntentional':      'Ball',
    'FoulBallFieldable':    'Foul',
    'HitByPitch':           'Hit By Pitch',
}
PITCH_TYPE_MARKERS = {
    'Fastball':  'o',
    'Curveball': 's',
    'Slider':    '^',
    'Changeup':  'D',
}
PLAY_RESULT_STYLES = {
    'Single':  ('blue',   'o'),
    'Double':  ('purple', 'o'),
    'Triple':  ('gold',   'o'),
    'HomeRun': ('orange', 'o'),
    'Out':     ('black',  'o'),
}

# ── Data helpers ───────────────────────────────────────────────────────────────

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file, low_memory=False)
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
    has_game_num = 'game_num' in df.columns and df['game_num'].notna().any()
    has_uid      = 'GameUID' in df.columns
    entries = []
    if has_game_num:
        id_cols = ['Date', 'game_num']
        if has_uid:                      id_cols.append('GameUID')
        if 'PitcherTeam' in df.columns: id_cols.append('PitcherTeam')
        reg = (
            df[df['game_num'].notna()]
            .drop_duplicates(subset=(['GameUID'] if has_uid else ['Date']))
            [id_cols].sort_values('game_num')
        )
        for _, row in reg.iterrows():
            opp   = row['PitcherTeam'] if 'PitcherTeam' in row.index else 'OPP'
            label = f"Game {int(row['game_num'])} vs {opp} ({row['Date']})"
            uid   = row['GameUID'] if has_uid else None
            entries.append((row['Date'], label, (row['Date'], uid)))
    reg_dates = {e[0] for e in entries}
    for d in df['Date'].unique():
        if d not in reg_dates:
            entries.append((d, d, (d, None)))
    entries.sort(key=lambda x: x[0])
    options = {}
    for _, label, value in entries:
        options[label] = value
    return options

# ── Branding helper ────────────────────────────────────────────────────────────

def draw_psu_header(fig, batter_name, game_label, logo_img=None, stats_line=None):
    """Draw navy header bar (14% of fig height) with logo, title, batter, stats."""
    # Header occupies top 14% of figure
    hax = fig.add_axes([0, 0.86, 1, 0.14])
    hax.set_xlim(0, 1)
    hax.set_ylim(0, 1)
    hax.axis('off')
    # Navy fill
    hax.add_patch(patches.Rectangle((0, 0), 1, 1, transform=hax.transAxes,
                                     facecolor=PSU_NAVY, clip_on=False))
    # Light blue bottom accent stripe
    hax.add_patch(patches.Rectangle((0, 0), 1, 0.04, transform=hax.transAxes,
                                     facecolor=PSU_LTBLUE, clip_on=False))

    if logo_img is not None:
        # logo aspect ratio: width/height. Axes coords: fig is 20w x 12h.
        logo_h_fig = 0.13          # height in figure fraction
        logo_aspect = logo_img.shape[1] / logo_img.shape[0]  # w/h pixels
        logo_w_fig = logo_h_fig * logo_aspect * (12 / 20)    # correct for fig aspect
        for x_pos in [0.002, 1.0 - logo_w_fig - 0.002]:
            lax = fig.add_axes([x_pos, 0.865, logo_w_fig, logo_h_fig])
            lax.imshow(logo_img)
            lax.axis('off')

    # Three text rows inside header
    hax.text(0.5, 0.75, 'PENN STATE NITTANY LIONS BASEBALL',
             ha='center', va='center', fontsize=18, fontweight='bold',
             color=PSU_WHITE, transform=hax.transAxes)
    hax.text(0.5, 0.46, f'{batter_name}   •   {game_label}',
             ha='center', va='center', fontsize=12,
             color=PSU_LTBLUE, transform=hax.transAxes)
    if stats_line:
        hax.text(0.5, 0.18, stats_line,
                 ha='center', va='center', fontsize=11,
                 color='#ccddee', transform=hax.transAxes)


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()

# ── Figure builders ────────────────────────────────────────────────────────────

def build_hitter_figures(batter_df, batter_name, game_label, logo_img=None):
    plate_appearance_groups = list(batter_df.groupby((batter_df['PitchofPA'] == 1).cumsum()))
    num_pa = len(plate_appearance_groups)

    # ── Pitch plot ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12), facecolor='white')
    gs  = GridSpec(3, 4, figure=fig,
                   width_ratios=[1.5, 1.5, 1.5, 1.2],
                   height_ratios=[1, 1, 1],
                   left=0.03, right=0.97, bottom=0.10, top=0.84,
                   wspace=0.25, hspace=0.40)

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
    sz_params     = {'x_start': -SZ_W_HALF,                         'y_start': 1.5,                      'width': 17/12,       'height': 3.3775-1.5}
    heart_params  = {'x_start': -SZ_W_HALF + (17/12)*0.33,         'y_start': 1.5+(3.3775-1.5)*0.33,   'width': (17/12)*0.34,'height': (3.3775-1.5)*0.34}
    shadow_params = {'x_start': -SZ_W_HALF - 0.2,                  'y_start': 1.3,                      'width': 17/12+0.4,  'height': 3.6-1.3}

    table_data = []

    for i, (pa_id, pa_data) in enumerate(plate_appearance_groups, start=1):
        if i > 9:
            break
        ax = axes[i - 1]
        pitcher_throws = pa_data.iloc[0]['PitcherThrows']
        handedness     = 'RHP' if pitcher_throws == 'Right' else 'LHP'
        pitcher_name   = pa_data.iloc[0]['Pitcher']
        ax.set_title(f'PA {i} vs {handedness}', fontsize=16, fontweight='bold', color=PSU_NAVY)
        ax.text(0.5, -0.13, f'P: {pitcher_name}', fontsize=10, fontstyle='italic',
                ha='center', transform=ax.transAxes, color='#555555')

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
                    f"{int(row['PitchofPA'])}", color='white', fontsize=9,
                    ha='center', va='center', weight='bold')
            pitch_speed = f"{round(row['RelSpeed'], 1)} MPH"
            pitch_type  = row['AutoPitchType']
            if row.name == pa_data.index[-1]:
                outcome_x = next(
                    (r for r in [row['PlayResult'], row['KorBB'], row['PitchCall']] if r != 'Undefined'),
                    'Undefined'
                )
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
               loc='lower center', bbox_to_anchor=(0.35, 0.01),
               ncol=len(unique_calls), fontsize=11, title_fontsize=12, frameon=False)
    fig.legend(handles=handles2, title='Pitch Type (Shapes)',
               loc='lower center', bbox_to_anchor=(0.35, -0.03),
               ncol=len(PITCH_TYPE_MARKERS), fontsize=11, title_fontsize=12, frameon=False)

    # Pitch-by-pitch table
    ax_table = fig.add_subplot(gs[:, 3])
    ax_table.axis('off')
    y_pos = 1.0
    for row in table_data:
        if 'PA' in row[0]:
            ax_table.text(0.05, y_pos, row[0], fontsize=13, fontweight='bold',
                          fontstyle='italic', color=PSU_NAVY)
            ax_table.axhline(y=y_pos - 0.01, color=PSU_NAVY, linewidth=1)
            y_pos -= 0.05
        else:
            ax_table.text(0.05, y_pos, f"  {row[0]}  |  {row[1]}  |  {row[2]}", fontsize=11)
            y_pos -= 0.04

    # Stats — passed into header so nothing overlaps the plot area
    whiffs    = batter_df['PitchCall'].eq('StrikeSwinging').sum()
    hard_hits = batter_df[(batter_df['PitchCall']=='InPlay') & (batter_df['ExitSpeed']>=95)].shape[0]
    barrels   = batter_df[(batter_df['ExitSpeed']>=95) & (batter_df['Angle'].between(10,35))].shape[0]
    swing_calls = ['Foul','InPlay','StrikeSwinging','FoulBallFieldable','FoulBallNotFieldable']
    swings    = batter_df[batter_df['PitchCall'].isin(swing_calls)]
    chase     = swings[
        (swings['PlateLocSide'] < -0.7083) | (swings['PlateLocSide'] > 0.7083) |
        (swings['PlateLocHeight'] < 1.5)   | (swings['PlateLocHeight'] > 3.3775)
    ].shape[0]
    stats_line = f"Whiffs: {whiffs}    Hard Hit: {hard_hits}    Barrels: {barrels}    Chase: {chase}"

    draw_psu_header(fig, batter_name, game_label, logo_img, stats_line=stats_line)

    # ── Batted ball figure ─────────────────────────────────────────────────────
    bb_fig = plt.figure(figsize=(20, 12), facecolor='white')
    bb_ax  = bb_fig.add_axes([0.04, 0.06, 0.55, 0.78])

    LF, LC, CF_dist, RC, RF = 330, 365, 390, 365, 330
    angles    = np.linspace(-45, 45, 500)
    distances = np.interp(angles, [-45,-30,0,30,45], [LF,LC,CF_dist,RC,RF])
    bb_ax.plot(distances*np.sin(np.radians(angles)), distances*np.cos(np.radians(angles)),
               color='black', linewidth=2)
    bb_ax.plot([-LF*np.sin(np.radians(45)),0],[LF*np.cos(np.radians(45)),0], color='black')
    bb_ax.plot([ RF*np.sin(np.radians(45)),0],[RF*np.cos(np.radians(45)),0], color='black')
    bb_ax.plot([0,90,0,-90,0],[0,90,180,90,0], color='brown', linewidth=2)

    for pa_number, pa_data in plate_appearance_groups:
        if pa_data.empty:
            continue
        last = pa_data.iloc[-1]
        if last['PitchCall'] != 'InPlay':
            continue
        if pd.isnull(last.get('Bearing')) or pd.isnull(last.get('Distance')):
            continue
        bearing  = np.radians(last['Bearing'])
        distance = last['Distance']
        ev       = round(last['ExitSpeed'], 1) if pd.notnull(last.get('ExitSpeed')) else 'NA'
        result   = last['PlayResult']
        x = distance * np.sin(bearing)
        y = distance * np.cos(bearing)
        color, marker = PLAY_RESULT_STYLES.get(result, ('black','o'))
        bb_ax.scatter(x, y, color=color, marker=marker, s=220, edgecolor='black', zorder=5)
        bb_ax.text(x, y, str(pa_number), color='white', fontsize=12,
                   fontweight='bold', ha='center', va='center', zorder=6)
        bb_ax.text(x, y-16, f"{ev} mph" if ev != 'NA' else 'NA',
                   color='red', fontsize=10, fontweight='bold', ha='center', zorder=6)

    bb_ax.set_xticks([]); bb_ax.set_yticks([])
    bb_ax.axis('equal')
    bb_ax.set_title(f'Batted Ball Locations — {batter_name}',
                    fontsize=18, fontweight='bold', color=PSU_NAVY, pad=12)

    legend_els = [
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='blue',   markersize=10, label='Single'),
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Double'),
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='gold',   markersize=10, label='Triple'),
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='HomeRun'),
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='black',  markersize=10, label='Out'),
    ]
    bb_fig.legend(handles=legend_els, loc='lower center', bbox_to_anchor=(0.32, 0.01),
                  ncol=5, fontsize=10, frameon=False)

    # Stats panel right side
    stats_ax = bb_fig.add_axes([0.62, 0.06, 0.35, 0.78])
    stats_ax.axis('off')
    stats_ax.set_xlim(0, 1); stats_ax.set_ylim(0, 1)
    stats_ax.add_patch(patches.FancyBboxPatch(
        (0.04, 0.04), 0.92, 0.92, boxstyle='round,pad=0.02',
        facecolor='#eef3f9', edgecolor=PSU_NAVY, linewidth=2.5
    ))
    # Header strip in box
    stats_ax.add_patch(patches.FancyBboxPatch(
        (0.04, 0.82), 0.92, 0.14, boxstyle='round,pad=0.01',
        facecolor=PSU_NAVY, edgecolor=PSU_NAVY, linewidth=0
    ))
    stats_ax.text(0.5, 0.895, 'GAME SUMMARY', ha='center', fontsize=14,
                  fontweight='bold', color=PSU_WHITE)

    stat_rows = [
        ('Pitches Seen',        len(batter_df)),
        ('Plate Appearances',   num_pa),
        ('Whiffs',              whiffs),
        ('Hard Hit (\u226595 mph)', hard_hits),
        ('Barrels',             barrels),
        ('Chase Swings',        chase),
    ]
    y_s = 0.755
    for label, val in stat_rows:
        stats_ax.text(0.12, y_s, label, fontsize=12, color='#333333', va='center')
        stats_ax.text(0.88, y_s, str(val), fontsize=14, fontweight='bold',
                      color=PSU_NAVY, ha='right', va='center')
        stats_ax.axhline(y_s - 0.055, xmin=0.08, xmax=0.92, color='#cccccc', linewidth=0.8)
        y_s -= 0.115

    draw_psu_header(bb_fig, batter_name, game_label, logo_img, stats_line=stats_line)

    return fig, bb_fig


def build_combined_pdf(batter_figures):
    """Combine all hitter figures into one landscape PDF."""
    pdf_buf = io.BytesIO()
    page_w, page_h = landscape(letter)
    c = rl_canvas.Canvas(pdf_buf, pagesize=(page_w, page_h))
    for batter_name, (pitch_fig, bb_fig) in batter_figures.items():
        for fig in [pitch_fig, bb_fig]:
            img_bytes  = fig_to_png_bytes(fig)
            img_reader = ImageReader(io.BytesIO(img_bytes))
            c.drawImage(img_reader, 0, 0, width=page_w, height=page_h,
                        preserveAspectRatio=True, anchor='c')
            c.showPage()
        plt.close(pitch_fig)
        plt.close(bb_fig)
    c.save()
    pdf_buf.seek(0)
    return pdf_buf.read()


# ── Streamlit UI ───────────────────────────────────────────────────────────────
st.title("Penn State Baseball \u2014 Postgame Hitter Report")

# Auto-load PSU logo bundled with the app
_logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'psu_logo.png (1)')
logo_img = mpimg.imread(_logo_path) if os.path.exists(_logo_path) else None

uploaded_csv = st.file_uploader(
    "\U0001f4c2 Upload Trackman CSV File",
    type=['csv'],
    help="Upload the Trackman export CSV for the game."
)

if not uploaded_csv:
    st.info("Upload a Trackman CSV file above to get started.")
    st.stop()

with st.spinner("Loading data..."):
    data = load_data(uploaded_csv)

if data.empty:
    st.error("No Penn State batter data found. Check that BatterTeam is 'PEN_NIT'.")
    st.stop()

game_options   = build_game_options(data)
game_labels    = list(game_options.keys())
selected_label = st.selectbox("Select a Game", options=game_labels, index=len(game_labels)-1)
selected_date, selected_uid = game_options[selected_label]

filtered_data = data[data['Date'] == selected_date]
if selected_uid is not None and 'GameUID' in filtered_data.columns:
    filtered_data = filtered_data[filtered_data['GameUID'] == selected_uid]

unique_batters = sorted(filtered_data['Batter'].dropna().unique())
if not unique_batters:
    st.warning("No batters found for the selected game.")
    st.stop()

st.markdown(f"**{len(unique_batters)} hitters found** for *{selected_label}*")
st.caption(', '.join(unique_batters))

# ── Preview ────────────────────────────────────────────────────────────────────
st.markdown('---')
st.subheader("Preview a Hitter")
preview_batter = st.selectbox("Select Hitter to Preview", options=unique_batters)
if preview_batter:
    batter_df = filtered_data[filtered_data['Batter'] == preview_batter]
    if not batter_df.empty:
        with st.spinner(f"Generating preview for {preview_batter}..."):
            p_fig, bb_fig = build_hitter_figures(batter_df, preview_batter, selected_label, logo_img)
        st.pyplot(p_fig, use_container_width=True)
        st.pyplot(bb_fig, use_container_width=True)
        plt.close('all')

# ── Export single combined PDF ─────────────────────────────────────────────────
st.markdown('---')
st.subheader("Export Full Report PDF")
st.write(
    "Generates **one PDF** containing every hitter — 2 pages per player "
    "(pitch plot + batted ball chart), in alphabetical order."
)

if st.button("\u2b07\ufe0f  Generate PDF for All Hitters", type="primary"):
    batter_figures = {}
    progress = st.progress(0, text="Building report\u2026")
    for idx, batter in enumerate(unique_batters):
        batter_df = filtered_data[filtered_data['Batter'] == batter]
        if batter_df.empty:
            continue
        p_fig, bb_fig = build_hitter_figures(batter_df, batter, selected_label, logo_img)
        batter_figures[batter] = (p_fig, bb_fig)
        progress.progress((idx + 1) / len(unique_batters),
                          text=f"Built {idx+1}/{len(unique_batters)}: {batter}")

    with st.spinner("Assembling PDF\u2026"):
        pdf_bytes = build_combined_pdf(batter_figures)

    game_tag = selected_date.replace('-', '')
    st.download_button(
        label="\U0001f4c4  Download Hitter Report PDF",
        data=pdf_bytes,
        file_name=f"PSU_HitterReport_{game_tag}.pdf",
        mime="application/pdf",
    )
    st.success(f"\u2705 {len(batter_figures)} hitters \u2014 {len(batter_figures)*2} pages total!")
