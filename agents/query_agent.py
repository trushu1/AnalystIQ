
import os
import json
import pandas as pd

from groq import Groq
from dotenv import load_dotenv


load_dotenv()


class NaturalLanguageQueryAgent:

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Please check your .env file."
            )

        self.client = Groq(api_key=api_key)
        self.model = "openai/gpt-oss-120b"

    # --------------------------------
    # STEP 1: INTERPRET QUESTION
    # --------------------------------

    def interpret_question(self, question, dataframe):

        columns = dataframe.columns.tolist()
        data_types = dataframe.dtypes.astype(str).to_dict()

        prompt = f"""
        You are an intelligent data analysis assistant.

        Convert the user's question into a JSON instruction.

        Available dataset columns:
        {columns}

        Column data types:
        {data_types}

        User question:
        {question}

        Supported operations:
        mean, sum, min, max, median, count_rows,
        value_counts, groupby_aggregate, top_n.

        Return ONLY a valid JSON object.

        Examples:

        Question: What is the average salary?
        JSON: {{"operation": "mean", "column": "Salary"}}

        Question: How many employees are there?
        JSON: {{"operation": "count_rows"}}

        Question: How many employees work in IT?
        JSON: {{"operation": "count_rows", "filter_column": "Department", "filter_value": "IT"}}

        Question: Show the top 3 highest-paid employees.
        JSON: {{"operation": "top_n", "column": "Salary", "n": 3, "ascending": false}}

        Question: Which department has the highest average salary?
        JSON: {{"operation": "groupby_aggregate", "group_column": "Department", "value_column": "Salary", "aggregation": "mean"}}

        Do not generate Python code or SQL.
        Do not invent column names.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Convert data questions into safe JSON instructions."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return json.loads(response.choices[0].message.content)

    # --------------------------------
    # STEP 2: VALIDATE AND EXECUTE
    # --------------------------------

    def execute_plan(self, plan, dataframe):

        operation = plan.get("operation")

        allowed_operations = [
            "mean", "sum", "min", "max", "median",
            "count_rows", "value_counts",
            "groupby_aggregate", "top_n"
        ]

        if operation not in allowed_operations:
            raise ValueError("Unsupported operation requested.")

        def check_column(column):
            if not isinstance(column, str) or column not in dataframe.columns:
                raise ValueError(f"Invalid column: {column}")

        if operation == "count_rows":

            filter_column = plan.get("filter_column")

            if filter_column:
                check_column(filter_column)

                filter_value = plan.get("filter_value")

                result = int(
                    (dataframe[filter_column].astype(str) ==
                     str(filter_value)).sum()
                )
            else:
                result = len(dataframe)

        elif operation in ["mean", "sum", "min", "max", "median"]:

            column = plan.get("column")
            check_column(column)

            if not pd.api.types.is_numeric_dtype(dataframe[column]):
                raise ValueError("This operation requires a numeric column.")

            value = getattr(dataframe[column], operation)()
            result = float(value) if pd.notna(value) else None

        elif operation == "value_counts":

            column = plan.get("column")
            check_column(column)

            result = dataframe[column].value_counts().to_dict()

        elif operation == "groupby_aggregate":

            group_column = plan.get("group_column")
            value_column = plan.get("value_column")
            aggregation = plan.get("aggregation")

            check_column(group_column)
            check_column(value_column)

            if aggregation not in ["mean", "sum", "min", "max", "median"]:
                raise ValueError("Unsupported aggregation.")

            if not pd.api.types.is_numeric_dtype(dataframe[value_column]):
                raise ValueError("The value column must be numeric.")

            result = (
                dataframe.groupby(group_column)[value_column]
                .agg(aggregation)
                .sort_values(ascending=False)
                .to_dict()
            )

        elif operation == "top_n":

            column = plan.get("column")
            check_column(column)

            if not pd.api.types.is_numeric_dtype(dataframe[column]):
                raise ValueError("The top N operation requires a numeric column.")

            n = plan.get("n", 5)

            if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 20:
                raise ValueError("Please request between 1 and 20 records.")

            ascending = plan.get("ascending", False)

            if not isinstance(ascending, bool):
                raise ValueError("Invalid sorting direction.")

            result = (
                dataframe.sort_values(
                    by=column,
                    ascending=ascending
                )
                .head(n)
            )

        return result

    # --------------------------------
    # STEP 3: EXPLAIN RESULT
    # --------------------------------

    def explain_result(self, question, result):

        if isinstance(result, pd.DataFrame):

            result_data = result.to_dict(orient="records")

            for row in result_data:
                for key, value in row.items():
                    if isinstance(value, float):
                        row[key] = round(value, 2)

        elif isinstance(result, dict):

            result_data = {
                key: round(value, 2) if isinstance(value, float) else value
                for key, value in result.items()
            }

        elif isinstance(result, float):

            result_data = round(result, 2)

        else:
            result_data = result

        prompt = f"""
        You are a professional data analyst.

        User question:
        {question}

        Calculated result:
        {json.dumps(result_data, default=str)}

        Explain the result in very simple English.

        Requirements:
        1. Directly answer the user's question.
        2. Highlight the most important result.
        3. Include relevant numerical values.
        4. Keep the explanation short and clear.
        5. Do not invent facts or calculations.
        6. Use only the supplied result.
        7. Do not assume or invent a currency symbol or unit.
           Use only the units explicitly provided in the dataset.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You explain data analysis results accurately."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        return response.choices[0].message.content

    # --------------------------------
    # STEP 4: COMPLETE QUERY WORKFLOW
    # --------------------------------

    def answer_question(self, question, dataframe):

        plan = self.interpret_question(question, dataframe)
        result = self.execute_plan(plan, dataframe)
        explanation = self.explain_result(question, result)

        return {
            "question": question,
            "plan": plan,
            "result": result,
            "explanation": explanation
        }
