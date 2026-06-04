import marimo

__generated_with = "0.17.8"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import pdfplumber
    import requests
    from io import BytesIO
    return BytesIO, mo, np, pd, pdfplumber, requests


@app.function
def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,  # Deletion
                           dp[i][j - 1] + 1,  # Insertion
                           dp[i - 1][j - 1] + cost) # Substitution

    return dp[m][n]


@app.cell
def _(BytesIO, np, pd, pdfplumber, requests):
    def load_canadian_renewables_data(pdf_location="https://renewablesassociation.ca/wp-content/uploads/2025/01/New-Project-List.pdf"):
        tech_labels = ['Wind', 'Solar', 'Energy Storage', "Solar-Storage", "Wind-Storage"]
        provinces = ['NL', 'PE' ,'NS', 'NB', 'QC', 'ON', 'MB', 'SK', 'AB', 'BC', 'YT', 'NT', 'NU']
        pdf_response = requests.get(pdf_location)
        def my_float(_x):
            try:
                _o = float(_x)
            except ValueError:
                _o = np.nan
            return _o
        def my_int(_x):
            try:
                _o = int(_x)
            except ValueError:
                _o = np.nan
            return _o
        all_tables = []
        with BytesIO(pdf_response.content) as pdf_file:
            with pdfplumber.open(pdf_file) as pdf:
                for _page in pdf.pages:
                    _tables = _page.extract_tables()
                    all_tables.append(_tables)
        columns = all_tables[0][0][0]
        data = pd.DataFrame(columns=columns)
        _ix = 0
        for _p in range(len(all_tables)):
            for _r in range(len(all_tables[_p][0])):
                if _p == 0 and _r == 0:
                    continue
                else:
                    _v = all_tables[_p][0][_r]
                    _label = _v[1]
                    _distances = [levenshtein_distance(_label, _t) for _t in tech_labels]
                    _new_label = tech_labels[np.argmin(_distances)]
                    _province = _v[2]
                    _distances = [levenshtein_distance(_province, _t) for _t in provinces]
                    _new_p = provinces[np.argmin(_distances)]
                    data.loc[_ix] = [_v[0], _new_label, _new_p, my_int(_v[3]), my_float(_v[4]), 
                                      my_float(_v[5]), my_float(_v[6]), 
                                      my_float(_v[7]), _v[8]]
                    _ix += 1
        return data
    return (load_canadian_renewables_data,)


@app.cell
def _(load_canadian_renewables_data):
    canadian_renewables = load_canadian_renewables_data()
    return (canadian_renewables,)


@app.cell
def _(canadian_renewables):
    canadian_renewables
    return


@app.cell
def _():
    pdf_location = "https://renewablesassociation.ca/wp-content/uploads/2025/01/New-Project-List.pdf"
    return (pdf_location,)


@app.cell
def _(pdf_location, requests):
    pdf_response = requests.get(pdf_location)
    return (pdf_response,)


@app.cell
def _(BytesIO, np, pd, pdf_response, pdfplumber):
    def my_float(_x):
        try:
            _o = float(_x)
        except ValueError:
            _o = np.nan
        return _o
    def my_int(_x):
        try:
            _o = int(_x)
        except ValueError:
            _o = np.nan
        return _o
    all_tables = []
    with BytesIO(pdf_response.content) as pdf_file:
        with pdfplumber.open(pdf_file) as pdf:
            for _page in pdf.pages:
                _tables = _page.extract_tables()
                all_tables.append(_tables)
    columns = all_tables[0][0][0]
    data = pd.DataFrame(columns=columns)
    _ix = 0
    for _p in range(len(all_tables)):
        for _r in range(len(all_tables[_p][0])):
            if _p == 0 and _r == 0:
                continue
            else:
                _v = all_tables[_p][0][_r]
                data.loc[_ix] = [_v[0], _v[1], _v[2], my_int(_v[3]), my_float(_v[4]), 
                                  my_float(_v[5]), my_float(_v[6]), 
                                  my_float(_v[7]), _v[8]]
                _ix += 1
    return (data,)


@app.cell
def _():
    return


@app.cell
def _(data):
    data
    return


@app.cell
def _(data):
    data.groupby('Technology')['Project Name'].count()
    return


@app.cell
def _(data):
    techs = list(set(data['Technology']))
    techs
    return (techs,)


@app.cell
def _(mo, np, techs):
    targets = ['Wind', 'Solar', 'Energy Storage', "Solar-Storage", "Wind-Storage"]
    test = techs[10]
    distances = [levenshtein_distance(test, _t) for _t in targets]
    _text = f"""
    - test string: {test}
    - closest match: {targets[np.argmin(distances)]}"""
    mo.md(_text)
    return


if __name__ == "__main__":
    app.run()
