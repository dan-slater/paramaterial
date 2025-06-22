from sqlalchemy.dialects.postgresql import UUID, JSONB
from .database import db, UUIDMixin, TimestampMixin

class Equipment(UUIDMixin, TimestampMixin, db.Model):
    __tablename__ = 'equipment'
    
    organization_id = db.Column(UUID(as_uuid=False), db.ForeignKey('organizations.id'), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    model = db.Column(db.String(200))
    description = db.Column(db.Text)
    equipment_type = db.Column(db.String(100), nullable=False)  # 'gleeble', 'sem', 'ebsd', 'tensile_tester', etc.
    specifications = db.Column(JSONB, default=dict)
    location = db.Column(db.String(200))
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    
    # Additional metadata
    manufacturer = db.Column(db.String(200))
    serial_number = db.Column(db.String(100))
    installation_date = db.Column(db.Date)
    last_calibration = db.Column(db.Date)
    next_calibration_due = db.Column(db.Date)
    
    # Relationships
    organization = db.relationship('Organization', back_populates='equipment')
    templates = db.relationship('AnalysisTemplate', back_populates='equipment')
    jobs = db.relationship('Job', back_populates='equipment')
    
    def get_template_count(self):
        """Get number of templates for this equipment"""
        return len([t for t in self.templates if t.is_public])
    
    def get_recent_jobs_count(self, days=30):
        """Get number of recent jobs using this equipment"""
        from datetime import datetime, timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        return len([j for j in self.jobs if j.created_at >= cutoff])
    
    @property
    def is_calibration_due(self):
        """Check if calibration is due"""
        if self.next_calibration_due:
            from datetime import date
            return date.today() >= self.next_calibration_due
        return False
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'organization_id': self.organization_id,
            'name': self.name,
            'model': self.model,
            'description': self.description,
            'equipment_type': self.equipment_type,
            'specifications': self.specifications,
            'location': self.location,
            'is_active': self.is_active,
            'manufacturer': self.manufacturer,
            'serial_number': self.serial_number,
            'installation_date': self.installation_date.isoformat() if self.installation_date else None,
            'last_calibration': self.last_calibration.isoformat() if self.last_calibration else None,
            'next_calibration_due': self.next_calibration_due.isoformat() if self.next_calibration_due else None,
            'is_calibration_due': self.is_calibration_due,
            'template_count': self.get_template_count(),
            'recent_jobs_count': self.get_recent_jobs_count(),
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<Equipment {self.name} ({self.equipment_type})>'

# Common equipment types for UCT Centre for Materials Engineering
EQUIPMENT_TYPES = {
    'gleeble': {
        'name': 'Gleeble Thermomechanical Simulator',
        'description': 'Thermomechanical testing and heat treatment',
        'typical_specs': {
            'max_temperature': 1200,  # Celsius
            'max_force': 100000,  # N
            'heating_rate': 10000  # C/s
        }
    },
    'sem': {
        'name': 'Scanning Electron Microscope',
        'description': 'High-resolution imaging and analysis',
        'typical_specs': {
            'resolution': 1.0,  # nm
            'magnification': 1000000,  # max magnification
            'acceleration_voltage': 30  # kV
        }
    },
    'ebsd': {
        'name': 'Electron Backscatter Diffraction',
        'description': 'Crystal orientation and texture analysis',
        'typical_specs': {
            'angular_resolution': 0.5,  # degrees
            'spatial_resolution': 20,  # nm
            'indexing_rate': 2000  # patterns/second
        }
    },
    'tensile_tester': {
        'name': 'Universal Testing Machine',
        'description': 'Mechanical property testing',
        'typical_specs': {
            'max_load': 100000,  # N
            'crosshead_speed': 500,  # mm/min
            'load_accuracy': 0.5  # %
        }
    },
    'hardness_tester': {
        'name': 'Hardness Tester',
        'description': 'Vickers, Brinell, Rockwell hardness testing',
        'typical_specs': {
            'load_range': '1-1000',  # kg
            'accuracy': 1,  # %
            'magnification': 400  # x
        }
    },
    'optical_microscope': {
        'name': 'Optical Microscope',
        'description': 'Microstructural examination',
        'typical_specs': {
            'magnification': 1000,  # max
            'resolution': 200,  # nm
            'objective_lenses': '5x,10x,20x,50x,100x'
        }
    }
}