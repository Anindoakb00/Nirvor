import networkx as nx
from typing import Dict, Any

class PolypharmacyGraph:
    """Deterministic, symbolic polypharmacy knowledge graph."""

    def __init__(self):
        self.graph = nx.Graph()
        self._build_graph()

    def _build_graph(self):
        # Adding some generic and known interacting pairs.
        # This can be expanded significantly.
        # Format: node1, node2, interaction_details
        interactions = [
            ("ketorolac", "aceclofenac", {
                "risk_level": "CRITICAL",
                "clinical_mechanism": "Concurrent Ketorolac and Aceclofenac administration causes cumulative COX inhibition and acute renal afferent arteriolar vasoconstriction.",
                "recommended_action": "Immediately discontinue one NSAID and monitor serum creatinine.",
                "bangla_patient_warning": "সতর্কতা: দুটি ব্যথার ওষুধ একসাথে সেবন করলে কিডনির মারাত্মক ক্ষতি হতে পারে। অবিলম্বে ডাক্তারের পরামর্শ নিন।"
            }),
            ("nsaid", "nsaid", {
                "risk_level": "CRITICAL",
                "clinical_mechanism": "Concurrent administration of multiple NSAIDs causes cumulative COX inhibition and acute renal afferent arteriolar vasoconstriction.",
                "recommended_action": "Immediately discontinue one NSAID and monitor serum creatinine.",
                "bangla_patient_warning": "সতর্কতা: দুটি ব্যথার ওষুধ একসাথে সেবন করলে কিডনির মারাত্মক ক্ষতি হতে পারে। অবিলম্বে ডাক্তারের পরামর্শ নিন।"
            }),
            # Add more hardcoded rules as needed
        ]

        for drug1, drug2, data in interactions:
            self.graph.add_edge(drug1.lower(), drug2.lower(), **data)

    def check_interaction(self, medications: list) -> Dict[str, Any]:
        """Checks for interactions among a list of medications (using generic names or categories)."""
        # Collect generic names and categories
        drug_entities = []
        for med in medications:
            generic_name = med.get("generic_name", "").lower()
            category = med.get("category", "").lower()

            if generic_name:
                drug_entities.append(generic_name)
            if category:
                # E.g., if category contains 'nsaid', add it as an entity to check
                if "nsaid" in category:
                    drug_entities.append("nsaid")

        # Check all pairs in the list
        for i in range(len(drug_entities)):
            for j in range(i + 1, len(drug_entities)):
                d1 = drug_entities[i]
                d2 = drug_entities[j]

                if self.graph.has_edge(d1, d2):
                    edge_data = self.graph.get_edge_data(d1, d2)
                    return {
                        "risk_level": edge_data["risk_level"],
                        "clinical_mechanism": edge_data["clinical_mechanism"],
                        "recommended_action": edge_data["recommended_action"],
                        "bangla_patient_warning": edge_data["bangla_patient_warning"],
                        "provider_used": "deterministic_graph"
                    }

        return {
            "risk_level": "LOW",
            "clinical_mechanism": "No significant interactions found.",
            "recommended_action": "Continue as prescribed.",
            "bangla_patient_warning": "কোনো সমস্যা পাওয়া যায়নি।",
            "provider_used": "deterministic_graph"
        }
