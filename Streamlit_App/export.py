import json
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from io import BytesIO

class Export:
  #def __init__(self, massages):

  def export_to_json(self,messages):
    return json.dumps(messages)
  
  def export_to_pdf(self, messages):
    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    flowables = []

    for i, message in enumerate(messages):
        if message['role'] == 'assistant':
          role_paragraph = Paragraph(f"Question: <b>{message['content']}:</b>", styles['Normal'])
          flowables.append(role_paragraph)
          flowables.append(Spacer(1, 12))
        else:
          message_paragraph = Paragraph(message['content'], styles['Normal'])
          flowables.append(message_paragraph)
          flowables.append(Spacer(1, 24))

    doc.build(flowables)
    buffer.seek(0)
    return buffer
  