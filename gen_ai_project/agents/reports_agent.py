# agents/reporting_agent.py

import logging
import os
import io
import traceback
from typing import Dict, Any, Optional, List

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateNotFound, TemplateSyntaxError # For HTML templating
from weasyprint import HTML, CSS # For HTML to PDF conversion

# --- LangChain Components ---
from langchain_core.prompts import ChatPromptTemplate, SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# --- Project Imports ---
from .base_agent import BaseAgent # Import the base class
from ..utils import config # To potentially get template paths or default styles

# Logger setup is handled by the BaseAgent's __init__
# We just need to use self.logger

class ReportingAgent(BaseAgent):
    """
    Generates PDF reports from data and natural language requests.
    Uses an LLM to structure content, Jinja2 for templating (optional),
    and WeasyPrint to render HTML to PDF.
    """
    def __init__(self, llm: Any, workspace_dir: str, verbose: bool = False):
        """
        Initializes the ReportingAgent.

        Args:
            llm: The language model instance.
            workspace_dir: The absolute path to the agent's workspace directory.
                           Required for saving reports, loading templates/data.
            verbose: If True, enable more detailed logging.
        """
        # Call BaseAgent's init first
        super().__init__(llm=llm, workspace_dir=workspace_dir, verbose=verbose)

        # Specific check: ReportingAgent needs a workspace
        if not self.workspace_dir:
            self.logger.critical("Initialization failed: ReportingAgent requires a valid workspace_dir.")
            raise ValueError("ReportingAgent requires a valid workspace_dir.")

        # Check for required libraries
        try:
            import jinja2
            import weasyprint
            self.logger.debug("Jinja2 and WeasyPrint libraries are available.")
        except ImportError as e:
             self.logger.critical(f"Missing required library: {e}. Please install Jinja2 and WeasyPrint (`pip install Jinja2 WeasyPrint` and follow WeasyPrint OS dependency setup).")
             raise ImportError(f"ReportingAgent requires Jinja2 and WeasyPrint: {e}")

        # Setup Jinja2 environment (look for templates in workspace/templates)
        template_dir = os.path.join(self.workspace_dir, "templates")
        os.makedirs(template_dir, exist_ok=True) # Ensure template dir exists
        try:
            self.jinja_env = Environment(
                loader=FileSystemLoader(template_dir),
                autoescape=select_autoescape(['html', 'xml']) # Basic security
            )
            self.logger.info(f"Jinja2 environment initialized, loading templates from: {template_dir}")
        except Exception as e:
             self.logger.error(f"Failed to initialize Jinja2 environment: {e}", exc_info=self.verbose)
             # Agent can potentially still function without templates using pure LLM generation
             self.jinja_env = None

        self.logger.info("ReportingAgent specific setup complete.")


    def _parse_report_request(self, request: str, data_source: str) -> Optional[Dict[str, Any]]:
        """Uses LLM to parse the request into a structured plan for the report."""
        self.logger.debug(f"Parsing report request: '{request[:100]}...' for data source '{data_source}'")
        parser = JsonOutputParser()
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
You are an expert report planner. Analyze the user's request for a PDF report based on the provided data source.
Extract the following information in JSON format:
- report_title: (string) A suitable title for the report based on the request.
- data_source_path: (string) The path/identifier of the main data source mentioned.
- output_filename: (string) The desired output filename for the PDF (e.g., 'report.pdf'). Ensure it ends with '.pdf'. If not specified, create a default like 'generated_report.pdf'.
- sections: (list of dicts) A list of sections required in the report. Each dict should have:
    - "type": (string) Type of section (e.g., 'summary_text', 'data_table', 'analysis_paragraph', 'image_placeholder').
    - "title": (string or null) Title for this section.
    - "content_description": (string) A detailed description for the LLM generating the content, explaining what data to use, what analysis to perform, or what text to write for this section. For 'data_table', specify columns or filtering. For 'image_placeholder', specify the expected image filename (relative to workspace).
- template_name: (string or null) If the user requests a specific template (e.g., 'monthly_summary_template.html'), specify its name. Otherwise, null.
- style_instructions: (string or null) Any specific styling requests (e.g., "use company colors", "make tables striped").

Example for 'sections':
[
  {"type": "summary_text", "title": "Executive Summary", "content_description": "Write a brief 2-paragraph summary of the key findings from the sales data."},
  {"type": "data_table", "title": "Sales Data Sample", "content_description": "Display the first 10 rows of the sales data, including columns 'Date', 'Region', 'Product', 'Amount'."},
  {"type": "image_placeholder", "title": "Sales Trend", "content_description": "Placeholder for the 'sales_trend.png' image."},
  {"type": "analysis_paragraph", "title": "Regional Performance", "content_description": "Analyze sales performance by region based on the data, highlighting top and bottom performers."}
]

Ensure the output is a valid JSON object. If crucial information (like data source or output filename logic) is missing, return an error structure: {"error": "Missing required info..."}.
"""),
            HumanMessage(content=f"Parse the following report request for data source '{data_source}':\n\n{request}")
        ])

        chain = prompt | self.llm | parser
        try:
            parsed_plan = chain.invoke({})
            self.logger.debug(f"LLM parsed report plan: {parsed_plan}")

            # --- Validation ---
            if not isinstance(parsed_plan, dict):
                 self.logger.warning("LLM report plan parsing did not return a dict.")
                 return {"error": "LLM parsing failed to return a dictionary."}
            if "error" in parsed_plan:
                 self.logger.warning(f"LLM parsing returned error: {parsed_plan['error']}")
                 return parsed_plan
            if not parsed_plan.get('report_title') or not parsed_plan.get('sections') or not isinstance(parsed_plan['sections'], list):
                 self.logger.warning("LLM failed to extract report_title or sections list.")
                 return {"error": "Could not determine report title or structure (sections) from the request."}
            if not parsed_plan.get('output_filename'):
                 parsed_plan['output_filename'] = "generated_report.pdf" # Default filename
                 self.logger.debug("Output filename not specified, using default 'generated_report.pdf'.")
            elif not parsed_plan['output_filename'].lower().endswith('.pdf'):
                 parsed_plan['output_filename'] += '.pdf' # Ensure .pdf extension
                 self.logger.debug(f"Added .pdf extension to output filename: {parsed_plan['output_filename']}")

            # Validate section structure minimally
            for i, section in enumerate(parsed_plan['sections']):
                 if not isinstance(section, dict) or "type" not in section or "content_description" not in section:
                      self.logger.warning(f"Invalid structure for section {i} in plan: {section}")
                      return {"error": f"Invalid structure for section {i} in the generated report plan."}

            return parsed_plan
        except Exception as e:
            self.logger.error(f"Error parsing report request with LLM: {e}", exc_info=self.verbose)
            return {"error": f"LLM parsing failed: {e}"}


    def _generate_html_content(self, plan: Dict[str, Any], df: Optional[pd.DataFrame]) -> Optional[str]:
        """Generates the HTML content for the report using the plan, data, and potentially Jinja2."""
        self.logger.info("Generating HTML content for the report...")
        template = None
        template_name = plan.get('template_name')

        # --- Try loading a Jinja template if specified and environment exists ---
        if template_name and self.jinja_env:
            try:
                template = self.jinja_env.get_template(template_name)
                self.logger.info(f"Using Jinja template: {template_name}")
            except TemplateNotFound:
                self.logger.warning(f"Jinja template '{template_name}' not found in {self.jinja_env.loader.searchpath}. Proceeding without template.")
            except TemplateSyntaxError as e:
                 self.logger.error(f"Syntax error in Jinja template '{template_name}': {e}. Proceeding without template.", exc_info=self.verbose)
            except Exception as e:
                 self.logger.error(f"Error loading Jinja template '{template_name}': {e}. Proceeding without template.", exc_info=self.verbose)

        # --- Prepare context for template or direct LLM generation ---
        report_context = {
            "report_title": plan.get("report_title", "Generated Report"),
            "sections": plan.get("sections", []),
            "style_instructions": plan.get("style_instructions"),
            "data_summary": df.describe().to_html(classes='dataframe table table-striped table-sm', border=0) if df is not None and not df.empty else "No data summary available.",
            "data_head": df.head().to_html(classes='dataframe table table-striped table-sm', border=0, index=False) if df is not None and not df.empty else "No data preview available.",
            # Add more pre-processed data snippets if useful
        }

        # --- Generate HTML ---
        html_content = None
        if template:
            # Render using Jinja template
            try:
                # We might need more sophisticated logic here to pass specific data parts
                # to the template based on the plan's sections.
                # For now, pass the whole plan and basic data summaries.
                html_content = template.render(report_context)
                self.logger.debug("HTML content generated using Jinja template.")
            except Exception as e:
                self.logger.error(f"Error rendering Jinja template '{template_name}': {e}", exc_info=self.verbose)
                # Fallback to direct LLM generation if template rendering fails?
                html_content = None # Ensure fallback if error occurs

        if not html_content:
            # Generate HTML directly using LLM if no template or template failed
            self.logger.info("Generating HTML content directly using LLM (no template used or template failed).")
            # Prepare data representation for LLM (e.g., head, describe, maybe column names)
            data_representation = "No data loaded."
            if df is not None:
                 data_representation = f"Data Columns: {df.columns.tolist()}\n\nData Head:\n{report_context['data_head']}\n\nData Summary:\n{report_context['data_summary']}"

            # Construct a detailed prompt for the LLM to generate HTML
            html_gen_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=f"""
You are an expert HTML report generator. Create a complete HTML document (including <!DOCTYPE html>, <html>, <head>, <body>) based on the provided report plan and data summary.
- Use the 'report_title' for the HTML `<title>` and a main `<h1>`.
- Iterate through the 'sections' list in the plan.
- For each section:
    - Add a section heading (e.g., `<h2>`) using the section 'title' if provided.
    - Generate content based on the section 'type' and 'content_description':
        - 'summary_text', 'analysis_paragraph': Write the requested text within `<p>` tags. Use the provided data summary/head for context.
        - 'data_table': Generate an HTML table (`<table>`, `<thead>`, `<tbody>`, `<tr>`, `<th>`, `<td>`) based on the 'content_description' (e.g., filtering columns, showing specific rows from the provided data head/summary). Use the provided data head as a reference for table structure. Add basic table classes like 'table table-striped'.
        - 'image_placeholder': Include an `<img>` tag. The 'content_description' should contain the relative image filename (e.g., 'plot.png'). Use that filename in the `src` attribute (e.g., `<img src="plot.png" alt="Section Title" style="max-width: 100%; height: auto;">`). Assume images are in the same directory as the final PDF or use relative paths correctly based on the workspace structure.
- Include basic CSS styling within a `<style>` tag in the `<head>` for readability (e.g., body font, table borders, padding). Incorporate any 'style_instructions' from the plan.
- Ensure the generated HTML is well-formed and complete.
- Base URL for relative paths (like images) will be the agent's workspace directory.
"""),
                HumanMessage(content=f"""
Report Plan:
```json
{json.dumps(plan, indent=2)}
