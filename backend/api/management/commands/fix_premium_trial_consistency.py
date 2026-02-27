from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from api.models import User


class Command(BaseCommand):
    help = "Corrige usuarios inconsistentes con plan Premium y billing free/sin trial."

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="No persiste cambios; solo muestra cuántos usuarios se corregirían.",
        )

    def handle(self, *args, **options):
        dry_run = bool(options.get("dry_run"))
        now = timezone.now()

        inconsistent_qs = User.objects.filter(plan="Premium").exclude(billing_status="active").filter(
            billing_status="free",
            trial_active=False,
        )

        updated = 0
        for user in inconsistent_qs.iterator():
            updated += 1
            if dry_run:
                continue
            user.trial_active = True
            user.trial_started_at = now
            user.trial_ends_at = now + timedelta(days=14)
            user.billing_status = "trial"
            user.save(update_fields=["trial_active", "trial_started_at", "trial_ends_at", "billing_status"])

        mode = "DRY-RUN" if dry_run else "APPLY"
        self.stdout.write(self.style.SUCCESS(f"{mode}: usuarios corregidos={updated}"))
