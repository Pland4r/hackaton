# controllers.py
from flask import jsonify, request
from bson.json_util import dumps # For converting MongoDB documents to JSON

# This function will be called from your main app to register all routes
def register_candidate_routes(app, db):
    """
    Registers all API routes for the candidates collection.
    
    Args:
        app: The Flask application instance.
        db: The MongoDB database connection.
    """
    candidates_collection = db.candidates
    
    # GET: List all candidates
    @app.route('/api/candidates', methods=['GET'])
    def list_candidates():
        candidates = candidates_collection.find()
        return dumps(candidates)

    # GET: Retrieve a single candidate by candidate_id
    @app.route('/api/candidate/<candidate_id>', methods=['GET'])
    def get_candidate(candidate_id):
        print('get condidate')
        candidate = candidates_collection.find_one({"candidate_id": candidate_id})
        if candidate:
            return dumps(candidate)
        return jsonify({"error": "Candidate not found"}), 404

    # POST: Create a new candidate
    @app.route('/api/candidates', methods=['POST'])
    def add_candidate():
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Insert the JSON data directly into the collection
        result = candidates_collection.insert_one(data)
        
        # Check if the insertion was successful
        if result.inserted_id:
            return jsonify({"message": "Candidate created", "id": str(result.inserted_id)}), 201
        
        return jsonify({"error": "Failed to create candidate"}), 500

    # PUT: Update an existing candidate
    @app.route('/api/candidate/<candidate_id>', methods=['PUT'])
    def update_candidate(candidate_id):
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Update the document using the $set operator
        result = candidates_collection.update_one(
            {"candidate_id": candidate_id},
            {"$set": data}
        )

        if result.matched_count == 0:
            return jsonify({"error": "Candidate not found"}), 404
        
        return jsonify({"message": "Candidate updated"})

    # DELETE: Delete a candidate
    @app.route('/api/candidate/<candidate_id>', methods=['DELETE'])
    def delete_candidate(candidate_id):
        result = candidates_collection.delete_one({"candidate_id": candidate_id})
        
        if result.deleted_count == 0:
            return jsonify({"error": "Candidate not found"}), 404
        
        return jsonify({"message": "Candidate deleted"})