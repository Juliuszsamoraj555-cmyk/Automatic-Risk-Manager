# translations.py

LANGS = {
    "PL": {
        "sidebar_title": "vAlpha Terminal",
        "login_info": "Zaloguj się, aby odblokować zaawansowane modele.",
        "btn_login": " Zaloguj / Rejestracja",
        "tab_login": "Logowanie",
        "tab_register": "Rejestracja",
        "tickers_label": "Symbole spółek (Tickery):",
        "tickers_help": "###  Jak wpisywać symbole?\nWprowadź listę aktywów...",
        "engine_label": "vAlpha Engine ",
        "engine_help": "Skorygowana symulacja Monte Carlo...",
        "email": "E-mail",
        "password": "Hasło",
        "msg_logged": "Zalogowano!",
        "msg_auth_error": "Błąd danych. Sprawdź e-mail i hasło.",
        "msg_fill_fields": "Wprowadź dane logowania.",
        "btn_register": "Załóż konto",
        "msg_reg_success": "Konto utworzone! Potwierdź e-mail, aby aktywować.",
        "msg_reg_error": "Błąd rejestracji. Hasło musi mieć min. 6 znaków.",
        "msg_fill_reg": "Uzupełnij pola rejestracji.",
        "welcome": "Witaj: **{}**",
        "status_pro": "STATUS: PRO ({} dni)",
        "status_free": "STATUS: FREE",
        "btn_unlock_pro": "ODBLOKUJ PRO",
        "btn_logout": "Wyloguj",
        "tickers_help": """
        ### 🔍 Jak wpisywać symbole?
        Wprowadź listę aktywów oddzieloną przecinkami. System pobiera dane z **Yahoo Finance**.
        

        **Przykłady dla rynków:**
        * 🇺🇸 **USA (Nasdaq/NYSE):** Sam symbol, np. `AAPL`, `MSFT`, `NVDA`.
        * 🇵🇱 **Polska (GPW):** Symbol z końcówką `.WA`, np. `PKO.WA`, `ALE.WA`, `CDR.WA`.
        * 🇪🇺 **Europa:** Końcówki `.DE` (Niemcy), `.PA` (Francja), `.L` (Londyn).
        * ₿ **Krypto:** Pary z USD, np. `BTC-USD`, `ETH-USD`.

        ---
        *Wskazówka: Jeśli nie znasz tickera, sprawdź go na finance.yahoo.com.*
        """,
        "currency": "PLN",
        "capital_label": "Kapitał ({}):",
        "tickers_label": "Symbole spółek (Tickery):",
        "opt_model_label": "Model Optymalizacji:",
        "opt_model_options": ["Bezpieczeństwo (VaR-First)", "Efektywność (Sortino)"],
        "opt_model_help": "VaR-First: Minimalizuje ryzyko nagłej straty. Sortino: Szuka najlepszego stosunku zysku do ryzyka.",
        "risk_label": "Ryzyko:",
        "risk_options": ["Niskie", "Średnie", "Wysokie"],
        "limit_2x_label": "Limit dywersyfikacji (2x)",
        "limit_2x_help": """
        ### Bezpiecznik portfela
        Sprawia, iż największa pozycja w portfelu może być maksymalnie dwukrotnie większa od najmniejszej pozycji. Ma to na celu ograniczyć ryzyko specyficzne.
        """,
        "min_weight_label": "Min. udział (PRO)",
        "min_weight_locked": " (ZABLOKOWANE)",
        "min_weight_help": """
        ### Minimalny udział
        Jeśli chcesz, aby konkretne aktywo stanowiło przynajmniej określoną część Twojego portfela, wpisz symbol i wartość, np. NVDA:10 (co oznacza minimum 10%).
        """,
        "run_mc_label": "Symulacje Monte Carlo",
        "run_mc_help": "Uruchamia 3000 losowych scenariuszy stóp zwrotu, aby sprawdzić, co może stać się z Twoim portfelem. Bez dodatkowych założeń jest ona bardzo teoretyczna.",
        "valpha_engine_label": "vAlpha Engine",
        "valpha_engine_locked": " (ZABLOKOWANE)",
        "valpha_engine_help": "Skorygowana symulacja Monte Carlo stosująca zamiast rozkładu normalnego Gaussa rozkład t-Studenta, opierająca się na modelu CAPM, Beta Decay oraz Alfie.",
        "expander_market": "PARAMETRY RYNKOWE",
        "rf_label": "Rf % (Risk-free):",
        "rf_help": "Stopa wolna od ryzyka (najczęściej przyjmowana jako rentowność 10-letnich obligacji skarbowych).",
        "rm_label": "Rm % (Oczekiwany zwrot rynku):",
        "rm_help": "Średni roczny zwrot z szerokiego indeksu (np. S&P 500).",
        "alpha_label": "Alfa % (Przewaga):",
        "alpha_help": "Ile % dotychczasowej przewagi utrzyma Twój portfel nad rynkiem.",
        "beta_speed_label": "Stabilizacja Bety:",
        "beta_speed_help": "Szybkość, z jaką Beta portfela dąży do średniej rynkowej w czasie symulacji. Im wyżej, tym szybciej portfel upodabnia się do rynku.",
        "msg_pro_required": "Ta funkcja wymaga konta PRO.",
        "btn_run_analysis": "URUCHOM ANALIZĘ",
        "err_free_limit": "Wersja FREE obsługuje do 5 spółek.",
        "tabs": ["Struktura Portfela", "Monte Carlo", "Korelacja", "Metodologia"],
        "t1_subheader": "Rekomendowana Alokacja",
        "metric_var_label": "Miesięczny VaR (95%)",
        "metric_var_caption": "Istnieje 95% prawdopodobieństwa, że miesięcznie Twój portfel nie straci więcej niż tę wartość procentową.",
        "metric_corr_label": "Średnia Korelacja",
        "metric_corr_caption": "Korelacja cenowa aktywów - im wyższa, tym mniejsza dywersyfikacja.",
        "metric_risk_val_label": "Ryzyko ({})",
        "metric_risk_val_caption": "Istnieje 95% prawdopodobieństwa, że miesięcznie Twój portfel nie straci więcej niż tę wartość.",
        "col_ticker": "Ticker",
        "col_share": "Udział (%)",
        "t4_header": "Metodologia vAlpha Engine",
        "t4_text": """
        Analiza portfela vAlpha opiera się na trzech filarach nowoczesnych finansów ilościowych:

        * **Model Ryzyka (Fat Tails):** W przeciwieństwie do standardowych modeli opartych na rozkładzie Gaussa, stosujemy **rozkład t-Studenta**. Pozwala to na uwzględnienie tzw. „grubych ogonów” (Fat Tails), czyli zjawiska, w którym ekstremalne krachy rynkowe zdarzają się częściej, niż przewiduje to klasyczna statystyka. Dzięki temu Twój $VaR$ (Value at Risk) jest bardziej realistyczny.
        * **Optymalizacja SLSQP:** Wykorzystujemy algorytm *Sequential Least Squares Programming*, aby znaleźć idealny punkt równowagi. System nie tylko szuka zysku, ale przede wszystkim rozwiązuje skomplikowane równanie matematyczne, które musi spełnić Twoje limity (np. limit dywersyfikacji 2x) przy jednoczesnej minimalizacji ryzyka.
        * **vAlpha Engine (Adjusted MC):** Nasza autorska symulacja Monte Carlo nie jest zwykłym „błądzeniem losowym”. Integruje ona model **CAPM** (Capital Asset Pricing Model), uwzględniając historyczną Alfę (Twoją przewagę) oraz Betę (wrażliwość na rynek). Dodatkowo stosujemy mechanizm **Beta Decay**, który zakłada, że ekstremalne wyniki spółek z czasem mają tendencję do stabilizowania się i dążenia w stronę średniej rynkowej.
        """,
        "err_analysis_main": "Błąd analizy: Proszę sprawdzić poprawność tickerów.",
        "err_details": "Szczegóły techniczne:",
        "mc_metrics": ["95. Percentyl", "3. Kwartyl (Q3)", "Mediana", "1. Kwartyl (Q1)", "5. Percentyl", "Prawd. straty", "CAGR"],
        "mc_5y": "PROGNOZA 5 LAT",
        "mc_10y": "PROGNOZA 10 LAT",
        "auth_popover": "Zaloguj się / Rejestracja",
        "spinner_loading": "Trwa analiza danych i symulacja Monte Carlo...",
        "err_generic": "Wystąpił nieoczekiwany błąd podczas analizy.",
        "err_details": "Szczegóły błędu",
        "tab_backtest": "Backtesting",
        "backtest_header": "Wyniki historyczne (Ostatnie 3 lata)",
        "backtest_desc": "Wykres pokazuje, jak zachowałby się ten portfel w przeszłości w porównaniu do indeksu S&P 500 (SPY).",
        "backtest_port_label": "Twój Portfel vAlpha",
        "backtest_bench_label": "Benchmark (S&P 500)",
        "error_no_data": "Błąd: Nie udało się pobrać danych dla podanych spółek. Sprawdź tickery.",
    
    },
    "EN": {
        "sidebar_title": "vAlpha Terminal",
        "login_info": "Please log in to unlock advanced models.",
        "btn_login": " Login / Register",
        "tab_login": "Login",
        "tab_register": "Register",
        "tickers_label": "Stock Symbols (Tickers):",
        "tickers_help": "###  How to enter symbols?\nEnter a list of assets...",
        "engine_label": "vAlpha Engine ",
        "engine_help": "Adjusted Monte Carlo simulation...",
        "email": "E-mail",
        "password": "Password",
        "msg_logged": "Logged in!",
        "msg_auth_error": "Authentication error. Check e-mail and password.",
        "msg_fill_fields": "Please enter your login details.",
        "btn_register": "Create account",
        "msg_reg_success": "Account created! Please confirm your email to activate.",
        "msg_reg_error": "Registration error. Password must be at least 6 chars.",
        "msg_fill_reg": "Please fill in all registration fields.",
        "welcome": "Welcome: **{}**",
        "status_pro": "STATUS: PRO ({} days)",
        "status_free": "STATUS: FREE",
        "btn_unlock_pro": "UNLOCK PRO",
        "btn_logout": "Logout",
        "tickers_help": """
        ### 🔍 How to enter symbols?
        Enter a comma-separated list of assets. The system fetches data from **Yahoo Finance**.

        **Market Examples:**
        * 🇺🇸 **USA (Nasdaq/NYSE):** Symbol only, e.g., `AAPL`, `MSFT`, `NVDA`.
        * 🇵🇱 **Poland (WSE):** Symbol with `.WA` suffix, e.g., `PKO.WA`, `ALE.WA`, `CDR.WA`.
        * 🇪🇺 **Europe:** Suffixes `.DE` (Germany), `.PA` (France), `.L` (London).
        * ₿ **Crypto:** USD pairs, e.g., `BTC-USD`, `ETH-USD`.

        ---
        *Tip: If you don't know the ticker, check it at finance.yahoo.com.*
        """,
        "currency": "USD",
        "capital_label": "Initial Capital ({}):",
        "tickers_label": "Stock symbols (Tickers):",
        "opt_model_label": "Optimization Model:",
        "opt_model_options": ["Safety (VaR-First)", "Efficiency (Sortino)"],
        "opt_model_help": "VaR-First: Minimizes the risk of sudden loss. Sortino: Seeks the best risk-adjusted return ratio.",
        "risk_label": "Risk Level:",
        "risk_options": ["Low", "Medium", "High"],
        "limit_2x_label": "Diversification Limit (2x)",
        "limit_2x_help": """
        ### Portfolio safety fuse
        Ensures that the largest position in the portfolio can be at most twice as large as the smallest position. This is intended to limit specific risk.
        """,
        "min_weight_label": "Min. Weight (PRO)",
        "min_weight_locked": " (LOCKED)",
        "min_weight_help": """
        ### Minimum Weight
        If you want a specific asset to represent at least a certain part of your portfolio, enter the ticker and value, e.g., NVDA:10 (meaning minimum 10%).
        """,
        "run_mc_label": "Monte Carlo Simulations",
        "run_mc_help": "Runs 3,000 random return scenarios to see what might happen to your portfolio. Without additional assumptions, it is highly theoretical.",
        "valpha_engine_label": "vAlpha Engine",
        "valpha_engine_locked": " (LOCKED)",
        "valpha_engine_help": "Adjusted Monte Carlo simulation using Student's t-distribution instead of Gaussian normal distribution, based on CAPM, Beta Decay, and Alpha models.",
        "expander_market": "MARKET PARAMETERS",
        "rf_label": "Rf % (Risk-free rate):",
        "rf_help": "Risk-free rate (usually represented by the yield of 10-year government bonds).",
        "rm_label": "Rm % (Expected market return):",
        "rm_help": "Average annual return of a broad market index (e.g., S&P 500).",
        "alpha_label": "Alpha % (Edge):",
        "alpha_help": "What percentage of your current outperformance the portfolio will maintain over the market.",
        "beta_speed_label": "Beta Stabilization:",
        "beta_speed_help": "The speed at which the portfolio's Beta gravitates toward the market average. Higher values mean faster alignment with the market.",
        "msg_pro_required": "This feature requires a PRO account.",
        "btn_run_analysis": "RUN ANALYSIS",
        "err_free_limit": "FREE version supports up to 5 assets.",
        "tabs": ["Portfolio Structure", "Monte Carlo", "Correlation", "Methodology"],
        "t1_subheader": "Recommended Allocation",
        "metric_var_label": "Monthly VaR (95%)",
        "metric_var_caption": "There is a 95% probability that your portfolio will not lose more than this percentage in a month.",
        "metric_corr_label": "Average Correlation",
        "metric_corr_caption": "Asset price correlation - the higher it is, the lower the diversification.",
        "metric_risk_val_label": "Risk ({})",
        "metric_risk_val_caption": "There is a 95% probability that your portfolio will not lose more than this amount in a month.",
        "col_ticker": "Ticker",
        "col_share": "Share (%)",
        "t4_header": "vAlpha Engine Methodology",
        "t4_text": """
        The vAlpha portfolio analysis is built upon three pillars of modern quantitative finance:

        * **Risk Model (Fat Tails):** Unlike standard models based on Gaussian distribution, we utilize the **Student's t-distribution**. This accounts for "Fat Tails"—the phenomenon where extreme market crashes occur more frequently than classical statistics would suggest. Consequently, your $VaR$ (Value at Risk) is much more robust and realistic.
        * **SLSQP Optimization:** We employ the *Sequential Least Squares Programming* algorithm to find the optimal equilibrium. The system doesn't just chase returns; it solves a complex mathematical equation that must satisfy your constraints (e.g., the 2x diversification limit) while simultaneously minimizing risk.
        * **vAlpha Engine (Adjusted MC):** Our proprietary Monte Carlo simulation is not a simple "random walk." It integrates the **CAPM** (Capital Asset Pricing Model), accounting for historical Alpha (your edge) and Beta (market sensitivity). Furthermore, we implement a **Beta Decay** mechanism, which assumes that extreme asset performances tend to stabilize and gravitate toward the market mean over time.
        """,
        "err_analysis_main": "Analysis error: Please verify that the tickers are correct.",
        "mc_metrics": ["95th Percentile", "3rd Quartile (Q3)", "Median", "1st Quartile (Q1)", "5th Percentile", "Prob. of loss", "CAGR"],
        "mc_5y": "5-YEAR FORECAST",
        "mc_10y": "10-YEAR FORECAST",
        "spinner_loading": "Analyzing data and running Monte Carlo simulation...",
        "auth_popover": "Login / Register",
        "err_generic": "An unexpected error occurred during the analysis.",
        "err_details": "Error details",
        "tab_backtest": "Backtesting",
        "backtest_header": "Historical Performance (Last 3 Years)",
        "backtest_desc": "The chart shows how this portfolio would have performed in the past compared to the S&P 500 index (SPY).",
        "backtest_port_label": "Your vAlpha Portfolio",
        "backtest_bench_label": "Benchmark (S&P 500)",
        "error_no_data": "Error: Could not fetch data for the given tickers. Please check the symbols.",
    }
}
