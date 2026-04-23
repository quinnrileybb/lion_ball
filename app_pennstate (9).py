# Streamlit PDF export fix (aspect ratio preserved)

from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.utils import ImageReader
import io

def save_fig_to_pdf(fig, pdf_canvas, x, y, max_width):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)

    img = ImageReader(buf)
    img_width, img_height = fig.get_size_inches()

    aspect = img_height / img_width
    draw_width = max_width
    draw_height = draw_width * aspect

    pdf_canvas.drawImage(img, x, y, width=draw_width, height=draw_height)

def create_pdf(figs, output_path="report.pdf"):
    c = rl_canvas.Canvas(output_path, pagesize=landscape(letter))
    page_width, page_height = landscape(letter)

    for fig in figs:
        save_fig_to_pdf(fig, c, x=40, y=40, max_width=page_width - 80)
        c.showPage()

    c.save()
