# ... end of the for loop ...
        minibatch = []

    # FIX: Add this block after the for loop to process any remaining rows
    if minibatch:
        logging.info(f"Processing the final remaining {len(minibatch)} rows.")
        # You might need to refresh the token one last time here as well
        (token_information_batch["token_start_time"],
         token_information_batch["current_token"],
        ) = check_and_refresh_token(...) # Pass the necessary args

        responses_json = process_row_minibatch(
            minibatch, token_information_batch["current_token"], app_config=app_config
        )
        for response_json in responses_json:
            target_file.write(json.dumps(response_json) + ",\n")
            target_file.flush()

    # Remove the last comma and close the array
    target_file.seek(target_file.tell() - 2, os.SEEK_SET)
    # ... rest of the function ...
