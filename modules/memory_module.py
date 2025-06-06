import os
from typing import Dict, List, Optional, Any
import json
from datetime import datetime

try:
    from mem0 import MemoryClient
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False

class InvestorAnalyzerMemory:
    """Memory layer for the Investor Analyzer using mem0 API"""
    
    def __init__(self):
        """Initialize memory system with API"""
        self.api_key = os.getenv("MEM0_API_KEY")
        self.user_id = "investor_analyzer_system"
        self.enabled = bool(self.api_key and MEM0_AVAILABLE)
        
        if self.enabled:
            try:
                self.client = MemoryClient()
                print("✅ Memory system initialized with mem0 SDK")
            except Exception as e:
                print(f"❌ Failed to initialize mem0 client: {e}")
                self.enabled = False
        else:
            if not MEM0_AVAILABLE:
                print("⚠️ mem0 SDK not installed - run: pip install mem0ai")
            if not self.api_key:
                print("⚠️ MEM0_API_KEY not found - memory features disabled")
    

    
    async def store_analysis_result(self, 
                                   analysis_data: Dict[str, Any], 
                                   company_info: Optional[Dict[str, str]] = None) -> bool:
        """Store analysis results in memory using API"""
        if not self.enabled:
            return False
            
        try:
            # Create structured memory content
            input_type = analysis_data.get('input_type', 'unknown')
            
            # Create comprehensive memory content for final analysis
            content = f"""COMPREHENSIVE ANALYSIS SESSION COMPLETE:

BUSINESS SUMMARY: {analysis_data.get('final_summary', 'Business analysis completed')}

KEY BUSINESS STRENGTHS: {analysis_data.get('key_strength', 'Multiple strengths identified')}

AREAS FOR IMPROVEMENT: {analysis_data.get('key_weakness', 'Several improvement areas noted')}

INVESTOR PERCEPTION: {analysis_data.get('investor_impression', 'Mixed investor engagement')}

MISSED OPPORTUNITIES: {analysis_data.get('missed_opportunity', 'Various optimization opportunities')}

ANALYSIS CONFIDENCE: {analysis_data.get('confidence_rating', 'Medium')}"""
            
            # Add investor feedback patterns if available
            if analysis_data.get("investor_response"):
                investor_data = analysis_data["investor_response"]
                if hasattr(investor_data, 'dict'):
                    investor_data = investor_data.dict()
                
                questions = investor_data.get('questions', [])
                signals = investor_data.get('interest_signals', [])
                objections = investor_data.get('objections', [])
                skepticism = investor_data.get('areas_of_skepticism', [])
                
                content += f"""

INVESTOR ENGAGEMENT ANALYSIS:
- Total Questions Asked: {len(questions)}
- Interest Signals Detected: {len(signals)}
- Objections Raised: {len(objections)}
- Areas of Skepticism: {len(skepticism)}

INVESTOR QUESTIONS:
{chr(10).join([f"• {q}" for q in questions[:3]]) if questions else "• No specific questions recorded"}

POSITIVE SIGNALS:
{chr(10).join([f"• {s}" for s in signals[:3]]) if signals else "• No positive signals recorded"}

CONCERNS & OBJECTIONS:
{chr(10).join([f"• {o}" for o in objections[:3]]) if objections else "• No major objections recorded"}

SKEPTICISM AREAS:
{chr(10).join([f"• {s}" for s in skepticism[:3]]) if skepticism else "• No skepticism areas identified"}"""

            # Store memory using SDK
            messages = [{"role": "user", "content": content}]
            result = self.client.add(messages, user_id=self.user_id)
            print(f"📝 Stored analysis result in mem0")
            return True
            
        except Exception as e:
            print(f"❌ Failed to store analysis result: {e}")
            return False
    
    def search_context(self, query: str, limit: int = 5) -> List[str]:
        """Search memory for contextual insights"""
        if not self.enabled:
            return []
            
        try:
            print(f"🔎 Executing mem0 search for: '{query[:50]}...'")
            # Search using SDK
            results = self.client.search(query, user_id=self.user_id, limit=limit)
            print(f"🔎 mem0 search returned: {type(results)} with {len(results) if isinstance(results, list) else 'unknown'} results")
            
            if isinstance(results, dict) and "results" in results:
                memories = results["results"]
                memory_texts = [memory.get("memory", "") for memory in memories if memory.get("memory")]
                print(f"🔎 Extracted {len(memory_texts)} memory texts")
                return memory_texts
            elif isinstance(results, list):
                memory_texts = [memory.get("memory", "") for memory in results if memory.get("memory")]
                print(f"🔎 Extracted {len(memory_texts)} memory texts from list")
                return memory_texts
            else:
                print(f"🔎 Unexpected results format: {results}")
                return []
            
        except Exception as e:
            print(f"❌ Failed to search memories: {e}")
            import traceback
            print(traceback.format_exc())
            return []
    
    async def get_contextual_insights(self, current_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get contextual insights based on historical data using mem0 API"""
        if not self.enabled:
            return {"insights": [], "comparisons": [], "recommendations": []}
        
        try:
            insights = []
            comparisons = []
            recommendations = []
            
            # Search for similar analyses based on key strength
            key_strength = current_analysis.get("key_strength", "")
            if key_strength:
                similar_memories = await self.search_context(f"Similar companies with strength: {key_strength}", limit=3)
                
                if similar_memories:
                    insights.append(f"Found {len(similar_memories)} similar analyses in memory")
                    comparisons.append("This pitch has commonalities with previously analyzed companies")
            
            # Search for investor feedback patterns
            if current_analysis.get("investor_response"):
                investor_data = current_analysis["investor_response"]
                if hasattr(investor_data, 'dict'):
                    investor_data = investor_data.dict()
                
                questions_count = len(investor_data.get("questions", []))
                signals_count = len(investor_data.get("interest_signals", []))
                
                feedback_memories = await self.search_context("investor questions and concerns", limit=3)
                
                if feedback_memories:
                    if signals_count > questions_count:
                        insights.append("Strong positive investor signals detected")
                    else:
                        recommendations.append("Consider addressing investor concerns more proactively")
            
            return {
                "insights": insights,
                "comparisons": comparisons,
                "recommendations": recommendations
            }
            
        except Exception as e:
            print(f"❌ Failed to get contextual insights: {e}")
            return {"insights": [], "comparisons": [], "recommendations": []}
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        if not self.enabled:
            return {"status": "disabled", "total_memories": 0}
            
        try:
            # Get stats using SDK
            try:
                memories = self.client.get_all(user_id=self.user_id)
                if isinstance(memories, dict) and "results" in memories:
                    total = len(memories["results"])
                else:
                    total = len(memories) if memories else 0
                
                return {
                    "status": "active",
                    "total_memories": total
                }
            except Exception as get_error:
                return {"status": "error", "error": f"Failed to get memories: {get_error}"}
            
        except Exception as e:
            print(f"❌ Failed to get memory stats: {e}")
            return {"status": "error", "error": str(e)}

# Global memory instance
memory_system = InvestorAnalyzerMemory() 