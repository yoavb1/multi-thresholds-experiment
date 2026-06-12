from django.db import models


class ExperimentData(models.Model):
    user_id = models.AutoField(primary_key=True)
    aid = models.CharField(max_length=255)
    csv_row_id = models.IntegerField(null=True, blank=True)  # Track which CSV row was assigned
    ps = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    human_sensitivity = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    ds_sensitivity = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)

    thresholds_distance = models.CharField(max_length=100, null=True, blank=True)
    architecture = models.CharField(max_length=100, null=True, blank=True)

    start_time = models.DateTimeField(auto_now_add=True)
    complete = models.BooleanField(default=False)
    end_time = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"User: {self.user_id}, Condition: {self.ds_sensitivity}, Started: {self.start_time}"

class ExperimentAction(models.Model):
    user_id = models.ForeignKey('ExperimentData', on_delete=models.CASCADE)
    block_number = models.IntegerField()
    trial_number = models.IntegerField()
    classification_decision = models.CharField(max_length=10, choices=[('signal', 'Signal'), ('noise', 'Noise')])
    stimulus_seen = models.FloatField()
    dss_judgment = models.CharField(max_length=10, choices=[('signal', 'Signal'), ('noise', 'Noise'), ('not sure', 'not Sure')])
    decision_time = models.FloatField()
    correct_classification = models.CharField(max_length=10, choices=[('signal', 'Signal'), ('noise', 'Noise')])

    class Meta:
        unique_together = ('user_id', 'block_number', 'trial_number')

    def __str__(self):
        return f"User: {self.user_id.user_id}, Block: {self.block_number}, Trial: {self.trial_number}"

class TOASTResponse(models.Model):
    user_id = models.ForeignKey('ExperimentData', on_delete=models.CASCADE)
    usefulness = models.IntegerField(null=True, blank=True)
    reliability = models.IntegerField(null=True, blank=True)
    trust = models.IntegerField(null=True, blank=True)
    confidence = models.IntegerField(null=True, blank=True)
    satisfaction = models.IntegerField(null=True, blank=True)
    predictability = models.IntegerField(null=True, blank=True)
    understandability = models.IntegerField(null=True, blank=True)
    surprised = models.IntegerField(null=True, blank=True)
    comfortable = models.IntegerField(null=True, blank=True)
    # Numeracy questions (1-6 scale)
    numeracy_fractions = models.IntegerField(null=True, blank=True)
    numeracy_shirt = models.IntegerField(null=True, blank=True)
    numeracy_useful = models.IntegerField(null=True, blank=True)
    # Demographics
    age_group = models.CharField(max_length=50, null=True, blank=True)
    gender = models.CharField(max_length=50, null=True, blank=True)
    education = models.CharField(max_length=100, null=True, blank=True)

    def __str__(self):
        return f"TOAST for User {self.user_id.user_id}"
