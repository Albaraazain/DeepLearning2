#!/usr/bin/env python3
"""
Browserless Form Automation Example
This script demonstrates how to use Browserless with Python for form filling and web automation.
"""

import requests
import json
import time
from typing import Dict, Any, Optional

class BrowserlessClient:
    """Client for interacting with Browserless API"""
    
    def __init__(self, base_url: str = "http://localhost:3000", token: str = "claude-browserless-2025"):
        self.base_url = base_url
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    
    def execute_function(self, code: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute JavaScript code using the Function API"""
        
        # Wrap the code in the correct format
        wrapped_code = f"""
        export default async function ({{ page, context }}) {{
            {code}
        }}
        """
        
        payload = {
            "code": wrapped_code,
            "context": context or {}
        }
        
        response = requests.post(
            f"{self.base_url}/function",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API call failed with status {response.status_code}: {response.text}")
    
    def fill_form_example(self, url: str, form_data: Dict[str, str]) -> Dict[str, Any]:
        """Example of filling a form on a webpage"""
        
        # JavaScript code to fill the form
        js_code = f"""
        // Navigate to the URL
        await page.goto('{url}', {{ waitUntil: 'networkidle2' }});
        
        // Wait for the page to load
        await page.waitForTimeout(2000);
        
        // Fill form fields
        const formData = {json.dumps(form_data)};
        
        for (const [selector, value] of Object.entries(formData)) {{
            try {{
                await page.waitForSelector(selector, {{ timeout: 5000 }});
                await page.type(selector, value);
                console.log(`Filled field ${{selector}} with value: ${{value}}`);
            }} catch (error) {{
                console.error(`Failed to fill field ${{selector}}: ${{error.message}}`);
            }}
        }}
        
        // Take a screenshot for verification
        const screenshot = await page.screenshot({{ encoding: 'base64' }});
        
        // Return results
        return {{
            success: true,
            screenshot: screenshot,
            url: page.url(),
            title: await page.title()
        }};
        """
        
        return self.execute_function(js_code)
    
    def advanced_form_automation(self, url: str) -> Dict[str, Any]:
        """Advanced form automation with stealth features"""
        
        js_code = f"""
        // Set stealth mode options
        await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36');
        
        // Navigate to the URL
        await page.goto('{url}', {{ 
            waitUntil: 'networkidle2',
            timeout: 30000
        }});
        
        // Human-like delays
        const humanDelay = () => Math.random() * 1000 + 500;
        
        // Check for common form elements
        const results = {{}};
        
        // Look for input fields
        const inputs = await page.$$('input[type="text"], input[type="email"], input[type="password"], textarea');
        results.inputFields = inputs.length;
        
        // Look for buttons
        const buttons = await page.$$('button, input[type="submit"]');
        results.buttons = buttons.length;
        
        // Look for select dropdowns
        const selects = await page.$$('select');
        results.selectFields = selects.length;
        
        // Check for potential captcha elements
        const captchaElements = await page.$$('[class*="captcha"], [id*="captcha"], iframe[src*="recaptcha"]');
        results.captchaDetected = captchaElements.length > 0;
        
        // Take screenshot
        const screenshot = await page.screenshot({{ encoding: 'base64' }});
        
        return {{
            success: true,
            screenshot: screenshot,
            url: page.url(),
            title: await page.title(),
            analysis: results
        }};
        """
        
        return self.execute_function(js_code)
    
    def handle_captcha_with_stealth(self, url: str) -> Dict[str, Any]:
        """Example of handling forms with potential captcha using stealth techniques"""
        
        js_code = f"""
        // Enhanced stealth settings
        await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36');
        await page.setViewport({{ width: 1366, height: 768 }});
        
        // Set additional headers to appear more human-like
        await page.setExtraHTTPHeaders({{
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        }});
        
        // Navigate with human-like behavior
        await page.goto('{url}', {{ 
            waitUntil: 'networkidle0',
            timeout: 30000
        }});
        
        // Random delay to mimic human behavior
        await page.waitForTimeout(Math.random() * 3000 + 2000);
        
        // Check page content and look for forms
        const pageInfo = {{
            title: await page.title(),
            url: page.url(),
            hasRecaptcha: false,
            hasHcaptcha: false,
            hasCloudflare: false,
            formElements: {{}}
        }};
        
        // Detect different types of protection
        try {{
            await page.waitForSelector('[src*="recaptcha"]', {{ timeout: 2000 }});
            pageInfo.hasRecaptcha = true;
        }} catch (e) {{}}
        
        try {{
            await page.waitForSelector('[src*="hcaptcha"]', {{ timeout: 2000 }});
            pageInfo.hasHcaptcha = true;
        }} catch (e) {{}}
        
        try {{
            const cfCheck = await page.$('[data-cf-beacon]') || await page.$('.cf-browser-verification');
            pageInfo.hasCloudflare = !!cfCheck;
        }} catch (e) {{}}
        
        // Analyze form elements
        const forms = await page.$$('form');
        for (let i = 0; i < forms.length; i++) {{
            const form = forms[i];
            const inputs = await form.$$('input');
            const buttons = await form.$$('button, input[type="submit"]');
            
            pageInfo.formElements[`form_${{i}}`] = {{
                inputs: inputs.length,
                buttons: buttons.length
            }};
        }}
        
        // Take screenshot for analysis
        const screenshot = await page.screenshot({{ 
            encoding: 'base64',
            fullPage: true 
        }});
        
        return {{
            success: true,
            screenshot: screenshot,
            pageInfo: pageInfo
        }};
        """
        
        return self.execute_function(js_code)


def main():
    """Main function to demonstrate usage"""
    client = BrowserlessClient()
    
    print("🚀 Browserless Form Automation Demo")
    print("=" * 50)
    
    # Example 1: Basic form analysis
    print("\n1. Analyzing a test form page...")
    try:
        result = client.advanced_form_automation("https://httpbin.org/forms/post")
        print(f"✅ Success! Found {result['analysis']['inputFields']} input fields")
        print(f"   Page title: {result['title']}")
        print(f"   Captcha detected: {result['analysis']['captchaDetected']}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 2: Stealth mode analysis
    print("\n2. Testing stealth capabilities on a protected site...")
    try:
        result = client.handle_captcha_with_stealth("https://www.google.com")
        print(f"✅ Success! Analyzed page: {result['pageInfo']['title']}")
        print(f"   Recaptcha detected: {result['pageInfo']['hasRecaptcha']}")
        print(f"   Cloudflare detected: {result['pageInfo']['hasCloudflare']}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n🎉 Demo completed!")
    print("You can now use these examples to build your own form automation scripts.")


if __name__ == "__main__":
    main()