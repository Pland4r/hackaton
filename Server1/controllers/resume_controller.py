from flask import jsonify
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.resume import ResumeManager
from typing import Dict, List

class ResumeController:
    def __init__(self):
        self.resume_manager = ResumeManager()
    
    def get_all_resumes(self) -> Dict:
        """Get all resume screening data"""
        try:
            resumes = self.resume_manager.get_all_resumes()
            return {
                'success': True,
                'data': resumes,
                'count': len(resumes),
                'message': 'Resume screening data retrieved successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to retrieve resume screening data'
            }
    
    def get_resumes_by_job_family(self, job_family: str) -> Dict:
        """Get resumes by job family"""
        try:
            resumes = self.resume_manager.get_resumes_by_job_family(job_family)
            return {
                'success': True,
                'data': resumes,
                'count': len(resumes),
                'message': f'Resumes in {job_family} family retrieved successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'Failed to retrieve resumes in {job_family} family'
            }
    
    def get_resumes_by_seniority(self, seniority: str) -> Dict:
        """Get resumes by seniority level"""
        try:
            resumes = self.resume_manager.get_resumes_by_seniority(seniority)
            return {
                'success': True,
                'data': resumes,
                'count': len(resumes),
                'message': f'{seniority} level resumes retrieved successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'Failed to retrieve {seniority} level resumes'
            }
    
    def get_advanced_resumes(self) -> Dict:
        """Get resumes that advanced to next stage"""
        try:
            resumes = self.resume_manager.get_advanced_resumes()
            return {
                'success': True,
                'data': resumes,
                'count': len(resumes),
                'message': 'Advanced resumes retrieved successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to retrieve advanced resumes'
            }
    
    def get_resume_stats(self) -> Dict:
        """Get resume statistics"""
        try:
            stats = self.resume_manager.get_resume_stats()
            return {
                'success': True,
                'data': stats,
                'message': 'Resume statistics retrieved successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to retrieve resume statistics'
            }
    
    def search_resumes(self, filters: Dict) -> Dict:
        """Search resumes with filters"""
        try:
            resumes = self.resume_manager.search_resumes(filters)
            return {
                'success': True,
                'data': resumes,
                'count': len(resumes),
                'filters_applied': filters,
                'message': 'Resume search completed successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to search resumes'
            }
    
    def get_skill_extraction(self) -> Dict:
        """Get skill extraction analysis"""
        try:
            analysis = self.resume_manager.get_skill_extraction()
            return {
                'success': True,
                'data': analysis,
                'message': 'Skill extraction analysis retrieved successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to retrieve skill extraction analysis'
            }
    
    def get_job_family_analysis(self) -> Dict:
        """Get job family analysis"""
        try:
            analysis = self.resume_manager.get_job_family_analysis()
            return {
                'success': True,
                'data': analysis,
                'message': 'Job family analysis retrieved successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to retrieve job family analysis'
            }