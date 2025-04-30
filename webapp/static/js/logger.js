/**
 * Paramaterial Console Logger
 * A comprehensive logging utility for browser debugging
 */

const ParaMaterialLogger = {
    // Log levels
    LEVELS: {
        DEBUG: 0,
        INFO: 1,
        WARNING: 2,
        ERROR: 3
    },
    
    // Current log level
    currentLevel: 0, // Default to DEBUG level
    
    // Enable or disable console output
    enabled: true,
    
    // History of log messages (limited to last 100)
    history: [],
    maxHistory: 100,
    
    /**
     * Initialize the logger
     * @param {Object} options Configuration options
     */
    init: function(options = {}) {
        if (options.level !== undefined) {
            this.currentLevel = options.level;
        }
        
        if (options.enabled !== undefined) {
            this.enabled = options.enabled;
        }
        
        if (options.maxHistory !== undefined) {
            this.maxHistory = options.maxHistory;
        }
        
        this.info('ParaMaterial Logger initialized', { level: this.getLevelName(this.currentLevel), enabled: this.enabled });
    },
    
    /**
     * Log a debug message
     * @param {string} message The message to log
     * @param {Object} data Optional data to include
     */
    debug: function(message, data = null) {
        this._log(this.LEVELS.DEBUG, message, data);
    },
    
    /**
     * Log an info message
     * @param {string} message The message to log
     * @param {Object} data Optional data to include
     */
    info: function(message, data = null) {
        this._log(this.LEVELS.INFO, message, data);
    },
    
    /**
     * Log a warning message
     * @param {string} message The message to log
     * @param {Object} data Optional data to include
     */
    warn: function(message, data = null) {
        this._log(this.LEVELS.WARNING, message, data);
    },
    
    /**
     * Log an error message
     * @param {string} message The message to log
     * @param {Object} data Optional data to include
     */
    error: function(message, data = null) {
        this._log(this.LEVELS.ERROR, message, data);
    },
    
    /**
     * Internal logging function
     * @private
     */
    _log: function(level, message, data) {
        if (!this.enabled || level < this.currentLevel) {
            return;
        }
        
        const timestamp = new Date().toISOString();
        const logEntry = {
            timestamp,
            level: this.getLevelName(level),
            message,
            data
        };
        
        // Add to history, keeping only the most recent entries
        this.history.push(logEntry);
        if (this.history.length > this.maxHistory) {
            this.history.shift();
        }
        
        // Format for console
        const formattedMessage = `[${timestamp}] [${logEntry.level}] ${message}`;
        
        // Output to console with appropriate styling
        switch (level) {
            case this.LEVELS.DEBUG:
                console.debug(formattedMessage, data || '');
                break;
            case this.LEVELS.INFO:
                console.info('%c' + formattedMessage, 'color: #3498db', data || '');
                break;
            case this.LEVELS.WARNING:
                console.warn('%c' + formattedMessage, 'color: #f39c12', data || '');
                break;
            case this.LEVELS.ERROR:
                console.error('%c' + formattedMessage, 'color: #e74c3c', data || '');
                break;
        }
    },
    
    /**
     * Get the string name of a log level
     * @param {number} level The level value
     * @returns {string} The level name
     */
    getLevelName: function(level) {
        const names = Object.keys(this.LEVELS);
        for (let name of names) {
            if (this.LEVELS[name] === level) {
                return name;
            }
        }
        return 'UNKNOWN';
    },
    
    /**
     * Get the log history
     * @returns {Array} The log history
     */
    getHistory: function() {
        return this.history;
    },
    
    /**
     * Clear the log history
     */
    clearHistory: function() {
        this.history = [];
        this.debug('Log history cleared');
    },
    
    /**
     * Add event listener instrumentation
     * @param {HTMLElement} element Element to instrument
     * @param {string} eventType Event type to listen for
     * @param {string} description Description of what's being monitored
     */
    monitorEvent: function(element, eventType, description) {
        if (!element) {
            this.error('Cannot monitor events on undefined element', { eventType, description });
            return;
        }
        
        element.addEventListener(eventType, (e) => {
            this.debug(`Event [${eventType}] - ${description}`, { 
                event: e.type,
                target: e.target.id || e.target.tagName,
                timestamp: new Date().toISOString()
            });
        });
        
        this.debug(`Monitoring ${eventType} events on ${element.id || element.tagName}`, { description });
    }
};

// Make the logger globally available
window.logger = ParaMaterialLogger;

// Initialize with default settings
document.addEventListener('DOMContentLoaded', function() {
    ParaMaterialLogger.init({
        level: ParaMaterialLogger.LEVELS.DEBUG,
        enabled: true
    });
});
