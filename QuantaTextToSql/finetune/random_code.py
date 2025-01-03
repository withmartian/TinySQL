# class CustomSFTTrainer(SFTTrainer):
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         labels = inputs.get("labels")
#         input_ids = inputs.get("input_ids")
#         attention_mask = inputs.get("attention_mask")

#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         logits = outputs.logits
#         ce_loss = outputs.loss

#         preds = torch.argmax(logits, dim=-1)
#         correct = (preds == labels) & (labels != -100)
#         accuracy = correct.sum().float() / (labels != -100).sum().float()
#         custom_loss = (1 - accuracy) * ce_loss

#         combined_loss = ce_loss + 0.1 * custom_loss

#         # Log individual losses and accuracy to W&B
#         wandb.log({
#             "Cross Entropy Loss": ce_loss.item(),
#             "Custom Loss": custom_loss.item(),
#             "Combined Loss": combined_loss.item(),
#             "Training Accuracy": accuracy.item(),
#         })

#         return (combined_loss, outputs) if return_outputs else combined_loss
    
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         import ipdb; ipdb.set_trace()
#         # Extract inputs
#         labels = inputs.get("labels")
#         input_ids = inputs.get("input_ids")
#         attention_mask = inputs.get("attention_mask")

#         # Forward pass through the model
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         logits = outputs.logits
#         ce_loss = outputs.loss  # Standard cross-entropy loss

#         # Generate SQL predictions
#         generated_ids = model.generate(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             max_new_tokens=100,  # Limit the length of generated tokens
#             temperature=0.7,
#             top_p=0.9,
#             pad_token_id=model.config.pad_token_id,
#             eos_token_id=model.config.eos_token_id,
#         )

#         # Decode predictions and ground truth SQL
#         prompts_length = input_ids.shape[1]
#         predicted_sqls = [
#             self.tokenizer.decode(gen_id[prompts_length:], skip_special_tokens=True).strip()
#             for gen_id in generated_ids
#         ]
#         ground_truth_sqls = [
#             self.tokenizer.decode(label[label != -100], skip_special_tokens=True).strip()[prompts_length:]
#             for label in labels
#         ]

#         # Compute evaluation scores using evaluate_cs_function
#         evaluation_scores = []
#         for pred_sql, gt_sql in zip(predicted_sqls, ground_truth_sqls):
#             item = dict_to_batchitem({"sql_statement": gt_sql})  # Convert to required format
#             score = self.evaluate_cs_function(item, pred_sql)  # Use the passed function
#             evaluation_scores.append(score)

#         # Convert scores to a tensor
#         evaluation_scores_tensor = torch.tensor(evaluation_scores, device=ce_loss.device)
        
#         # Calculate mean evaluation loss (penalty for deviation)
#         evaluation_loss = 1.0 - evaluation_scores_tensor.mean()  # Lower score implies higher penalty

#         # Combine CE loss and evaluation loss
#         combined_loss = ce_loss + 0.1 * evaluation_loss  # Scale evaluation loss as needed

#         return (combined_loss, outputs) if return_outputs else combined_loss