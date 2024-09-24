def create_batches(data_frame, batch_size):
    """
    creating batches to cover the complete dataset
    @param: data_frame
    @param: batch_size
    @return: batches
    """

    num_samples = len(data_frame)
    num_batches = num_samples // batch_size

    batches_data = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size
        batch = data_frame[start_idx:end_idx]
        batches_data.append(batch)

    remaining_samples = num_samples % batch_size
    if remaining_samples > 0:
        batch = data_frame[-remaining_samples:]
        batches_data.append(batch)

    return batches_data
