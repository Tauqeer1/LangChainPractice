from langchain_ollama import OllamaLLM


# Google model: gemma3:1b-it-q4_K_M

# Mistral model: mistral:7b-instruct-q3_K_L

# DeepSeek model: deepseek-r1:1.5b


"""
Task: Handle a very long input.

Description: Provide a very long text (e.g., a short article) as input to the model and ask it to summarize.

Objective: See how the model handles longer contexts (be mindful of context window limits of your chosen Ollama model).

"""


llm = OllamaLLM(model="gemma3:1b-it-q4_K_M")


prompt_text = """
China's economic ascent over the past few decades has been nothing short of remarkable, transforming it from an agrarian society to a global economic powerhouse. This incredible growth story, often termed an "economic miracle," began in earnest with the "Reform and Opening Up" policies initiated by Deng Xiaoping in 1978.

These reforms initially focused on de-collectivization of agriculture, significantly boosting food production and rural incomes. Concurrently, China began opening its doors to foreign investment, establishing Special Economic Zones that attracted capital, technology, and managerial expertise. This influx of foreign direct investment (FDI) played a crucial role in modernizing China's industrial base.

The gradual transition involved privatizing and contracting out many state-owned industries, fostering a more dynamic and competitive market. China's accession to the World Trade Organization (WTO) in 2001 further cemented its integration into the global economy, leading to a surge in exports and establishing it as the "world's factory."

Industrialization and manufacturing, particularly in electronics, textiles, and later automotive sectors, became key drivers of GDP growth. Massive infrastructure development, including an extensive high-speed rail network and modern ports, facilitated trade and internal connectivity. Urbanization also played a significant role, as large-scale migration from rural areas provided a continuous supply of relatively inexpensive labor for expanding industries.

The government's strategic focus on human capital development, with substantial investments in education and vocational training, created a more skilled workforce. A strong work ethic and cultural emphasis on education also contributed to productivity gains.

For decades, China maintained average annual GDP growth rates often exceeding 9-10%. This robust expansion lifted hundreds of millions out of poverty, creating a substantial middle class that fueled domestic consumption. Companies like Alibaba and JD.com revolutionized e-commerce, further stimulating internal demand.

China's growth has profoundly impacted the global economy. It became a voracious consumer of raw materials, benefiting commodity-exporting nations. Its manufacturing prowess provided affordable goods worldwide, contributing to global supply chains and disinflationary pressures. China's economic stability and development have become increasingly important for global growth, especially for Asian, African, and Latin American countries that trade heavily with it.

However, this rapid growth has not been without its challenges. Mounting debt, particularly in the real estate sector, poses a significant risk to financial stability. Sluggish consumer spending remains a concern, with a "dual-speed economy" evident in strong industrial output but weaker domestic demand. An aging population and shrinking workforce present long-term demographic challenges.

Environmental degradation, a consequence of rapid industrialization, requires substantial investment in clean energy and pollution control. Trade tensions, particularly with the United States, and concerns over intellectual property rights also add headwinds. China is actively working to shift towards a more consumption-driven and innovation-led growth model, investing heavily in high-tech sectors like AI, electric vehicles, and semiconductors. Despite facing these complex structural issues, China's economy continues to demonstrate resilience, with its strategic policies aiming to ensure sustained and high-quality development in the years to come.
"""

result = llm.invoke(f"Summarize this in 5 lines: {prompt_text}")

print(result)
