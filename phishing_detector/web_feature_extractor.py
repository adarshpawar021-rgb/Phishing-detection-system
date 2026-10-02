import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse


def analyze_webpage(url):
    """
    Download a webpage safely and extract basic webpage features.
    """

    # Default values
    features = {
        "LineOfCode": 0,
        "LargestLineLength": 0,
        "HasTitle": 0,
        "DomainTitleMatchScore": 0,
        "URLTitleMatchScore": 0,
        "HasFavicon": 0,
        "Robots": 0,
        "IsResponsive": 0,
        "NoOfURLRedirect": 0,
        "NoOfSelfRedirect": 0,
        "HasDescription": 0,
        "NoofPopup": 0,
        "NoOfiFrame": 0,
        "HasExternalFormSubmit": 0,
        "HasSocialNet": 0,
        "HasSubmitButton": 0,
        "HasHiddenFields": 0,
        "HasPasswordField": 0,
        "Bank": 0,
        "Pay": 0,
        "Crypto": 0,
        "HasCopyrightInfo": 0,
        "NoOfImage": 0,
        "NoOfCSS": 0,
        "NoOfJS": 0,
        "NoOfSelfRef": 0,
        "NoOfEmptyRef": 0,
        "NoOfExternalRef": 0
    }

    try:
        # Add scheme if missing
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()

        # --------------------------------------------------
        # Download webpage
        # --------------------------------------------------

        response = requests.get(
            url,
            timeout=5,
            allow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        html = response.text

        # --------------------------------------------------
        # Basic HTML features
        # --------------------------------------------------

        lines = html.splitlines()

        features["LineOfCode"] = len(lines)

        if lines:
            features["LargestLineLength"] = max(
                len(line) for line in lines
            )

        # Parse HTML
        soup = BeautifulSoup(
            html,
            "html.parser"
        )

        # --------------------------------------------------
        # Title
        # --------------------------------------------------

        title = soup.find("title")

        if title:
            title_text = title.get_text(
                strip=True
            ).lower()

            if title_text:
                features["HasTitle"] = 1

                # Domain-title similarity
                domain_words = (
                    domain
                    .replace(".", " ")
                    .replace("-", " ")
                    .split()
                )

                if domain_words:
                    matches = sum(
                        word in title_text
                        for word in domain_words
                        if len(word) > 2
                    )

                    features[
                        "DomainTitleMatchScore"
                    ] = matches / len(domain_words)

                # URL-title similarity
                url_words = (
                    url.lower()
                    .replace("/", " ")
                    .replace("-", " ")
                    .replace(".", " ")
                    .split()
                )

                if url_words:
                    matches = sum(
                        word in title_text
                        for word in url_words
                        if len(word) > 2
                    )

                    features[
                        "URLTitleMatchScore"
                    ] = matches / len(url_words)

        # --------------------------------------------------
        # Favicon
        # --------------------------------------------------

        favicon = soup.find(
            "link",
            rel=lambda value:
            value and
            "icon" in str(value).lower()
        )

        if favicon:
            features["HasFavicon"] = 1

        # --------------------------------------------------
        # Robots.txt
        # --------------------------------------------------

        try:
            robots_url = (
                f"{parsed_url.scheme}://"
                f"{parsed_url.netloc}/robots.txt"
            )

            robots_response = requests.get(
                robots_url,
                timeout=3,
                headers={
                    "User-Agent": "Mozilla/5.0"
                }
            )

            if robots_response.status_code == 200:
                features["Robots"] = 1

        except Exception:
            pass

        # --------------------------------------------------
        # Responsive design
        # --------------------------------------------------

        viewport = soup.find(
            "meta",
            attrs={
                "name": "viewport"
            }
        )

        if viewport:
            features["IsResponsive"] = 1

        # --------------------------------------------------
        # Description
        # --------------------------------------------------

        description = soup.find(
            "meta",
            attrs={
                "name": "description"
            }
        )

        if description:
            features["HasDescription"] = 1

        # --------------------------------------------------
        # iFrames
        # --------------------------------------------------

        features["NoOfiFrame"] = len(
            soup.find_all("iframe")
        )

        # --------------------------------------------------
        # Forms
        # --------------------------------------------------

        forms = soup.find_all("form")

        for form in forms:

            action = form.get(
                "action",
                ""
            ).strip()

            # External form submission
            if action.startswith(
                ("http://", "https://")
            ):

                action_domain = urlparse(
                    action
                ).netloc.lower()

                if (
                    action_domain
                    and action_domain != domain
                ):
                    features[
                        "HasExternalFormSubmit"
                    ] = 1

            # Submit buttons
            submit_buttons = form.find_all(
                ["input", "button"]
            )

            for button in submit_buttons:

                button_type = button.get(
                    "type",
                    ""
                ).lower()

                if button_type in (
                    "submit",
                    "button"
                ):
                    features[
                        "HasSubmitButton"
                    ] = 1

            # Hidden fields
            hidden_fields = form.find_all(
                "input",
                attrs={
                    "type": "hidden"
                }
            )

            if hidden_fields:
                features[
                    "HasHiddenFields"
                ] = 1

            # Password fields
            password_fields = form.find_all(
                "input",
                attrs={
                    "type": "password"
                }
            )

            if password_fields:
                features[
                    "HasPasswordField"
                ] = 1

        # --------------------------------------------------
        # Social media
        # --------------------------------------------------

        social_sites = [
            "facebook.com",
            "twitter.com",
            "instagram.com",
            "linkedin.com",
            "youtube.com"
        ]

        html_lower = html.lower()

        if any(
            site in html_lower
            for site in social_sites
        ):
            features["HasSocialNet"] = 1

        # --------------------------------------------------
        # Images
        # --------------------------------------------------

        features["NoOfImage"] = len(
            soup.find_all("img")
        )

        # --------------------------------------------------
        # CSS
        # --------------------------------------------------

        css_count = 0

        for link in soup.find_all(
            "link",
            href=True
        ):

            href = link.get(
                "href",
                ""
            ).lower()

            if ".css" in href:
                css_count += 1

        features["NoOfCSS"] = css_count

        # --------------------------------------------------
        # JavaScript
        # --------------------------------------------------

        features["NoOfJS"] = len(
            soup.find_all("script")
        )

        # --------------------------------------------------
        # Links / References
        # --------------------------------------------------

        links = soup.find_all(
            "a",
            href=True
        )

        for link in links:

            href = link.get(
                "href",
                ""
            ).strip()

            # Empty reference
            if not href:
                features[
                    "NoOfEmptyRef"
                ] += 1

                continue

            # Self reference
            if href.startswith("#"):
                features[
                    "NoOfSelfRef"
                ] += 1

                continue

            # External or self URL
            if href.startswith(
                ("http://", "https://")
            ):

                link_domain = urlparse(
                    href
                ).netloc.lower()

                if link_domain == domain:
                    features[
                        "NoOfSelfRef"
                    ] += 1
                else:
                    features[
                        "NoOfExternalRef"
                    ] += 1

        # --------------------------------------------------
        # Copyright
        # --------------------------------------------------

        if "copyright" in html_lower:
            features[
                "HasCopyrightInfo"
            ] = 1

        # --------------------------------------------------
        # Redirects
        # --------------------------------------------------

        features[
            "NoOfURLRedirect"
        ] = len(response.history)

        for redirect in response.history:

            previous_domain = urlparse(
                redirect.url
            ).netloc.lower()

            final_domain = urlparse(
                response.url
            ).netloc.lower()

            if previous_domain == final_domain:
                features[
                    "NoOfSelfRedirect"
                ] += 1

        # --------------------------------------------------
        # Keyword features
        # --------------------------------------------------

        if any(
            word in html_lower
            for word in [
                "bank",
                "banking"
            ]
        ):
            features["Bank"] = 1

        if any(
            word in html_lower
            for word in [
                "payment",
                "paypal",
                "pay now"
            ]
        ):
            features["Pay"] = 1

        if any(
            word in html_lower
            for word in [
                "bitcoin",
                "cryptocurrency",
                "crypto",
                "wallet"
            ]
        ):
            features["Crypto"] = 1

        return features

    except Exception as error:

        print(
            "\nWebpage analysis failed:"
        )

        print(error)

        print(
            "\nReturning default webpage features."
        )

        return features


# ==========================================================
# TEST PROGRAM
# ==========================================================

if __name__ == "__main__":

    print(
        "======================================"
    )

    print(
        "      WEBPAGE FEATURE EXTRACTOR"
    )

    print(
        "======================================"
    )

    test_url = input(
        "\nEnter URL to analyze: "
    )

    result = analyze_webpage(
        test_url
    )

    print(
        "\nWebpage Features:"
    )

    print(
        "--------------------------------------"
    )

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )

    print(
        "--------------------------------------"
    )

    print(
        "\nFeature extraction completed!"
    )