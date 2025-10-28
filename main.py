from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


load_dotenv()

def main():

    information = """
       Nicolas Sarközy de Nagy-Bocsa[d], dit Nicolas Sarkozy (par prononciation orthographique /ni.kɔ.la saʁ.kɔ.zi/[e] Écouterⓘ ; originellement Sárközy ou Sárközi [ˈʃaːɾkøzi], prononcé en hongrois « Charkeuzy »[3],[4],[5]), né le 28 janvier 1955 à Paris 17e (Seine), est un homme d'État français. Il est président de la République française du 16 mai 2007 au 15 mai 2012.

        Il occupe d'abord les fonctions de maire de Neuilly-sur-Seine, député, ministre du Budget et porte-parole du gouvernement ou encore de président par intérim du Rassemblement pour la République (RPR). À partir de 2002, il est ministre de l'Intérieur (à deux reprises), ministre de l'Économie et des Finances et président du conseil général des Hauts-de-Seine. Il est alors l'un des dirigeants les plus en vue de l'Union pour un mouvement populaire (UMP), qu'il préside de 2004 à 2007.

        Élu président de la République française en 2007 avec 53,1 % des suffrages face à Ségolène Royal, il inaugure une rupture de style et de communication par rapport à ses prédécesseurs. Il fait voter plusieurs réformes, dont celles des universités en 2007 et des retraites en 2010. Son mandat est également marqué par de grands événements internationaux tels que la crise économique mondiale de 2008, la crise de la dette dans la zone euro et l'intervention militaire de 2011 en Libye. Candidat à sa réélection en 2012 alors qu'il est au cœur de soupçons de financements illégaux de sa campagne électorale de 2007 par Liliane Bettencourt ou la Libye, il obtient 48,4 % des voix au second tour, s’inclinant face à François Hollande.

        Après son départ de la présidence, il siège pendant quelques mois au Conseil constitutionnel, dont il est membre de droit et à vie. En 2014, il reprend la présidence de l'Union pour un mouvement populaire (UMP), qu'il fait rebaptiser Les Républicains (LR). Il quitte la tête du parti en 2016 pour se présenter, sans succès, à la primaire de la droite et du centre en vue de l'élection présidentielle de 2017. Il se met ensuite de nouveau en retrait de la vie politique.

        En 2023, dans l'affaire Sarkozy-Azibert, il est condamné en appel à trois ans de prison pour corruption et trafic d'influence. Dans l'affaire Bygmalion, il est condamné en appel en 2024 à un an de prison pour financement illégal de sa campagne électorale de 2012 ; il est le premier président de la Ve République condamné à de la prison ferme, pour corruption et trafic d'influence.

        Dans l'affaire Sarkozy-Kadhafi, il est accusé d'avoir été corrompu par le dictateur libyen Mouammar Kadhafi en échange du financement illégal de sa campagne électorale, et il est condamné en première instance en septembre 2025 à 5 ans de prison ferme pour association de malfaiteurs. Il est écroué le 21 octobre à la prison de la Santé de Paris.
    """

    summary_template = """
        Given the {information} about a person I want you create:

        1. A short summary
        2. Two interresting fun facts
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    #llm = ChatOpenAI(model="gpt-5", temperature=0.5)
    llm = ChatOllama(model="gpt-oss:20b", temperature=0.5)
    chain = summary_prompt_template | llm

    response = chain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
